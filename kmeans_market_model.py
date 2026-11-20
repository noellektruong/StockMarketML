from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

# =====================================
# SETTINGS
# =====================================

DATA_FILE = Path("alpaca_hsy_mcd.csv")
OUTPUT_FILE = Path("kmeans_clustered_data.csv")
GRAPH_FILE = Path("kmeans_market_clusters.png")
MODEL_FILE = Path("kmeans_models.joblib")

STOCKS = ["HSY", "MCD"]
NUM_CLUSTERS = 3
RANDOM_STATE = 42

# Strict chronological blocks:
# 50% fit K-means/scalers; 15% discover candidate regimes;
# 15% validate candidates; final 20% is untouched until final evaluation.
MODEL_FIT_RATIO = 0.50
DISCOVERY_END_RATIO = 0.65
VALIDATION_END_RATIO = 0.80

STOCK_FEATURES = [
    "return_pct",
    "range_pct",
    "log_volume_ratio",
    "volatility_1h",
]

COMBINED_FEATURES = [
    "avg_return_pct",
    "return_spread_pct",
    "avg_range_pct",
    "avg_log_volume_ratio",
    "avg_volatility_1h",
    "volatility_spread",
]

FLAT_RETURN_THRESHOLD_PCT = 0.01


def load_and_create_features() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {DATA_FILE}. Run fetch_data.py first.")

    data = pd.read_csv(DATA_FILE)
    required = {"symbol", "ts_utc", "open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    data["ts_utc"] = pd.to_datetime(data["ts_utc"], utc=True, errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[data["symbol"].isin(STOCKS)].dropna(
        subset=["ts_utc", "open", "high", "low", "close", "volume"]
    ).copy()
    data = data.sort_values(["symbol", "ts_utc"]).reset_index(drop=True)

    found = set(data["symbol"].unique())
    if found != set(STOCKS):
        raise ValueError(f"Expected stocks {STOCKS}; found {sorted(found)}")

    print("\nStocks loaded:")
    print(data["symbol"].value_counts().to_string())
    print("\nCreating individual stock indicators...")

    by_stock = data.groupby("symbol", sort=False)
    data["return_pct"] = by_stock["close"].pct_change() * 100.0
    data["range_pct"] = (data["high"] - data["low"]) / data["close"] * 100.0

    rolling_volume = data.groupby("symbol")["volume"].transform(
        lambda values: values.rolling(12, min_periods=12).median()
    )
    volume_ratio = data["volume"] / rolling_volume.replace(0, np.nan)
    data["log_volume_ratio"] = np.log1p(volume_ratio.clip(lower=0))
    data["volatility_1h"] = data.groupby("symbol")["return_pct"].transform(
        lambda values: values.rolling(12, min_periods=12).std()
    )

    # Evaluation column only; it is never used as a K-means feature.
    data["forward_return_pct"] = (
        data.groupby("symbol")["close"].shift(-1) / data["close"] - 1.0
    ) * 100.0

    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=STOCK_FEATURES + ["forward_return_pct"]
    ).copy()
    if data.empty:
        raise ValueError("No rows remain after calculating indicators.")
    return data


def create_combined_features(data: pd.DataFrame) -> pd.DataFrame:
    value_cols = STOCK_FEATURES + ["forward_return_pct"]
    wide = data.pivot(index="ts_utc", columns="symbol", values=value_cols)
    wide.columns = [f"{feature}_{symbol}" for feature, symbol in wide.columns]
    wide = wide.dropna().copy()
    if wide.empty:
        raise ValueError("No synchronized HSY/MCD observations with complete features.")

    combined = pd.DataFrame(index=wide.index)
    combined["avg_return_pct"] = wide[["return_pct_HSY", "return_pct_MCD"]].mean(axis=1)
    combined["return_spread_pct"] = wide["return_pct_HSY"] - wide["return_pct_MCD"]
    combined["avg_range_pct"] = wide[["range_pct_HSY", "range_pct_MCD"]].mean(axis=1)
    combined["avg_log_volume_ratio"] = wide[
        ["log_volume_ratio_HSY", "log_volume_ratio_MCD"]
    ].mean(axis=1)
    combined["avg_volatility_1h"] = wide[
        ["volatility_1h_HSY", "volatility_1h_MCD"]
    ].mean(axis=1)
    combined["volatility_spread"] = (
        wide["volatility_1h_HSY"] - wide["volatility_1h_MCD"]
    ).abs()
    combined["avg_forward_return_pct"] = wide[
        ["forward_return_pct_HSY", "forward_return_pct_MCD"]
    ].mean(axis=1)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
    return combined


def assign_periods(frame: pd.DataFrame, cutoffs: dict) -> pd.Series:
    times = frame["ts_utc"]
    return pd.Series(
        np.select(
            [
                times < cutoffs["fit_cutoff"],
                times < cutoffs["discovery_cutoff"],
                times < cutoffs["validation_cutoff"],
            ],
            ["fit", "discovery", "validation"],
            default="test",
        ),
        index=frame.index,
        dtype="object",
    )


def market_mood(avg_return: float, avg_volatility: float, median_volatility: float) -> str:
    if avg_return > FLAT_RETURN_THRESHOLD_PCT:
        direction = "Positive-return"
    elif avg_return < -FLAT_RETURN_THRESHOLD_PCT:
        direction = "Negative-return"
    else:
        direction = "Sideways"
    volatility = "Higher-volatility" if avg_volatility >= median_volatility else "Lower-volatility"
    return f"{direction} / {volatility}"


def fit_and_score(
    frame: pd.DataFrame,
    features: list[str],
    return_column: str,
    volatility_column: str,
    cutoffs: dict,
    model_name: str,
):
    scored = frame.copy()
    scored["data_split"] = assign_periods(scored, cutoffs)
    fit_rows = scored[scored["data_split"] == "fit"].copy()
    if len(fit_rows) < NUM_CLUSTERS:
        raise ValueError(f"Not enough model-fit rows for {model_name}.")

    for period in ["fit", "discovery", "validation", "test"]:
        if (scored["data_split"] == period).sum() == 0:
            raise ValueError(f"The {period} period is empty for {model_name}.")

    if not np.isfinite(fit_rows[features].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite feature values in {model_name} model-fit data.")
    if not np.isfinite(scored[features].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite feature values in {model_name} data.")

    # Scaler and centroids are fitted ONLY on the earliest 50% block.
    scaler = RobustScaler()
    X_fit = scaler.fit_transform(fit_rows[features])
    model = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    model.fit(X_fit)

    centers = scaler.inverse_transform(model.cluster_centers_)
    return_idx = features.index(return_column)
    volatility_idx = features.index(volatility_column)
    raw_order = sorted(
        range(NUM_CLUSTERS),
        key=lambda cid: (centers[cid, return_idx], centers[cid, volatility_idx]),
    )
    raw_to_ordered = {raw_id: ordered_id for ordered_id, raw_id in enumerate(raw_order)}
    raw_all = model.predict(scaler.transform(scored[features]))
    scored["cluster"] = [raw_to_ordered[int(value)] for value in raw_all]

    fit_scored = scored[scored["data_split"] == "fit"]
    summary = fit_scored.groupby("cluster").agg(
        observations=("cluster", "size"),
        average_return=(return_column, "mean"),
        average_volatility=(volatility_column, "mean"),
    )
    median_vol = float(summary["average_volatility"].median())
    summary["market_mood"] = [
        market_mood(row["average_return"], row["average_volatility"], median_vol)
        for _, row in summary.iterrows()
    ]
    mood_map = summary["market_mood"].to_dict()
    scored["mood"] = scored["cluster"].map(mood_map)

    evaluation_col = "forward_return_pct" if "forward_return_pct" in scored else "avg_forward_return_pct"
    period_summaries = {}
    for period in ["fit", "discovery", "validation", "test"]:
        part = scored[scored["data_split"] == period]
        period_summaries[period] = part.groupby("cluster").agg(
            observations=("cluster", "size"),
            average_return=(return_column, "mean"),
            average_volatility=(volatility_column, "mean"),
            average_next_bar_return=(evaluation_col, "mean"),
        )

    package = {
        "model": model,
        "scaler": scaler,
        "features": list(features),
        "raw_to_ordered_cluster": raw_to_ordered,
        "mood_by_cluster": mood_map,
        "model_fit_through_utc": str(fit_rows["ts_utc"].max()),
        "split_cutoffs_utc": {key: str(value) for key, value in cutoffs.items()},
        "cluster_ordering": "Ordered by model-fit centroid return feature, lowest to highest.",
    }

    print(f"\n{model_name.upper()} — MODEL-FIT CLUSTERS")
    print("=" * (len(model_name) + 28))
    print(summary.round(5).to_string())
    for period, label in [("discovery", "STRATEGY DISCOVERY"), ("validation", "STRATEGY VALIDATION"), ("test", "UNTOUCHED TEST")]:
        print(f"\n{model_name.upper()} — {label}")
        print("=" * (len(model_name) + len(label) + 3))
        print(period_summaries[period].round(5).to_string())

    return scored, summary, period_summaries, package


def make_graph(stock_results: dict, combined_result: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    specs = [
        (axes[0], stock_results["HSY"], "return_pct", "volatility_1h", "HSY regimes"),
        (axes[1], stock_results["MCD"], "return_pct", "volatility_1h", "MCD regimes"),
        (axes[2], combined_result, "avg_return_pct", "avg_volatility_1h", "Combined HSY + MCD regimes"),
    ]
    for ax, frame, xcol, ycol, title in specs:
        test = frame[frame["data_split"] == "test"]
        for cluster_id in range(NUM_CLUSTERS):
            points = test[test["cluster"] == cluster_id]
            moods = points["mood"].dropna()
            mood_label = moods.iloc[0] if not moods.empty else "Unlabeled"
            ax.scatter(points[xcol], points[ycol], label=f"Cluster {cluster_id}: {mood_label}", alpha=0.5, s=12)
        ax.set_title(title)
        ax.set_xlabel("Five-minute return (%)")
        ax.set_ylabel("One-hour rolling volatility (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("K-means Market Regimes — Untouched Test Data", fontsize=14)
    fig.tight_layout()
    fig.savefig(GRAPH_FILE, dpi=250)
    plt.close(fig)
    print(f"\nSaved graph: {GRAPH_FILE}")


def main():
    print("\nK-MEANS MARKET REGIME MODEL — STRICT CHRONOLOGICAL SPLIT")
    print("=========================================================")
    print("Model fitting: first 50% | Discovery: next 15% | Validation: next 15% | Test: final 20%")
    print("Scalers and K-means centroids are fit only on the first 50%.")

    data = load_and_create_features()
    all_times = np.sort(data["ts_utc"].unique())
    if len(all_times) < 20:
        raise ValueError("Not enough unique timestamps for four chronological blocks.")
    cutoffs = {
        "fit_cutoff": pd.Timestamp(all_times[int(len(all_times) * MODEL_FIT_RATIO)]),
        "discovery_cutoff": pd.Timestamp(all_times[int(len(all_times) * DISCOVERY_END_RATIO)]),
        "validation_cutoff": pd.Timestamp(all_times[int(len(all_times) * VALIDATION_END_RATIO)]),
    }
    if not (cutoffs["fit_cutoff"] < cutoffs["discovery_cutoff"] < cutoffs["validation_cutoff"]):
        raise ValueError("Chronological cutoffs are not strictly increasing.")

    print("\nChronological cutoffs (UTC):")
    for label, value in cutoffs.items():
        print(f"{label}: {value}")

    stock_results = {}
    packages = {"stocks": {}, "combined": None}
    output_parts = []

    for stock in STOCKS:
        frame = data[data["symbol"] == stock].copy()
        scored, _, _, package = fit_and_score(
            frame, STOCK_FEATURES, "return_pct", "volatility_1h", cutoffs, stock
        )
        scored = scored.rename(columns={"cluster": "individual_cluster", "mood": "individual_mood"})
        stock_results[stock] = scored.rename(
            columns={"individual_cluster": "cluster", "individual_mood": "mood"}
        )
        packages["stocks"][stock] = package
        output_parts.append(scored)

    combined = create_combined_features(data)
    combined_scored, _, _, combined_package = fit_and_score(
        combined, COMBINED_FEATURES, "avg_return_pct", "avg_volatility_1h", cutoffs, "Combined HSY + MCD"
    )
    packages["combined"] = combined_package

    combined_merge = combined_scored[["ts_utc", "cluster", "mood", "data_split"]].rename(
        columns={
            "cluster": "combined_cluster",
            "mood": "combined_mood",
            "data_split": "combined_data_split",
        }
    )
    individual = pd.concat(output_parts, ignore_index=True)
    results = individual.merge(combined_merge, on="ts_utc", how="left", validate="many_to_one")
    results = results.sort_values(["symbol", "ts_utc"]).reset_index(drop=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved clustered observations: {OUTPUT_FILE}")

    packages["settings"] = {
        "stocks": STOCKS,
        "num_clusters": NUM_CLUSTERS,
        "stock_features": STOCK_FEATURES,
        "combined_features": COMBINED_FEATURES,
        "model_fit_ratio": MODEL_FIT_RATIO,
        "discovery_end_ratio": DISCOVERY_END_RATIO,
        "validation_end_ratio": VALIDATION_END_RATIO,
        "split_cutoffs_utc": {key: str(value) for key, value in cutoffs.items()},
        "random_state": RANDOM_STATE,
        "note": "Research and paper simulation only. No orders are placed.",
    }
    joblib.dump(packages, MODEL_FILE)
    print(f"Saved fitted models and scalers: {MODEL_FILE}")
    make_graph(stock_results, combined_scored)

    print("\nMODEL RUN COMPLETE")
    print("==================")
    print("Model-fit rows: first 50% of timestamps")
    print("Strategy discovery: next 15%")
    print("Strategy validation: next 15%")
    print("Untouched test: final 20%")
    print("Individual models: HSY, MCD")
    print("Combined model: HSY + MCD")
    print("Rows scored:", len(results))
    print("Combined synchronized timestamps:", len(combined_scored))
    print("No trades or orders were placed.")


if __name__ == "__main__":
    main()
