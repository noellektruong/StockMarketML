
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

# =====================================
# SETTINGS
# =====================================

DATA_FILE = "alpaca_hsy_mcd.csv"

OUTPUT_FILE = "kmeans_clustered_data.csv"
GRAPH_FILE = "kmeans_market_clusters.png"

STOCKS = ["HSY", "MCD"]
NUM_CLUSTERS = 3
TRAIN_RATIO = 0.80

FEATURES = [
    "return_pct",
    "range_pct",
    "log_volume_ratio",
    "volatility_1h"
]


def main():

    # =====================================
    # 1. LOAD DATA
    # =====================================

    print("\nLoading stock data...")

    data = pd.read_csv(DATA_FILE)

    data["ts_utc"] = pd.to_datetime(
        data["ts_utc"], utc=True
    )

    data = data[
        data["symbol"].isin(STOCKS)
    ].copy()

    data = data.sort_values(
        ["symbol", "ts_utc"]
    ).reset_index(drop=True)

    print("\nStocks loaded:")
    print(data["symbol"].value_counts())

    # =====================================
    # 2. CREATE MARKET INDICATORS
    # =====================================

    print("\nCreating market indicators...")

    # Percentage return from previous bar
    data["return_pct"] = (
        data.groupby("symbol")["close"]
        .pct_change() * 100
    )

    # Intraday price range relative to close
    data["range_pct"] = (
        (data["high"] - data["low"])
        / data["close"] * 100
    )

    # Compare volume with recent median volume
    rolling_volume = (
        data.groupby("symbol")["volume"]
        .transform(
            lambda x: x.rolling(
                12, min_periods=12
            ).median()
        )
    )

    volume_ratio = (
        data["volume"] / rolling_volume
    )

    data["log_volume_ratio"] = np.log1p(
        volume_ratio.clip(lower=0)
    )

    # Standard deviation of recent returns
    data["volatility_1h"] = (
        data.groupby("symbol")["return_pct"]
        .transform(
            lambda x: x.rolling(
                12, min_periods=12
            ).std()
        )
    )

    # Next five-minute bar return.
    # Used to evaluate clusters, NOT as a model feature.
    data["forward_return_pct"] = (
        data.groupby("symbol")["close"]
        .shift(-1)
        / data["close"] - 1
    ) * 100

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        subset=FEATURES + ["forward_return_pct"]
    ).copy()

    # =====================================
    # 3. CHRONOLOGICAL TRAIN / TEST SPLIT
    # =====================================

    print("\nSplitting training and testing data...")

    unique_times = np.sort(
        data["ts_utc"].unique()
    )

    cutoff_index = int(
        len(unique_times) * TRAIN_RATIO
    )

    cutoff_date = unique_times[cutoff_index]

    train = data[
        data["ts_utc"] < cutoff_date
    ].copy()

    test = data[
        data["ts_utc"] >= cutoff_date
    ].copy()

    if len(train) < NUM_CLUSTERS:
        raise ValueError("Not enough training observations.")

    print("Training observations:", len(train))
    print("Testing observations:", len(test))
    print("Training ends before:", cutoff_date)

    # =====================================
    # 4. SCALE FEATURES
    # =====================================

    print("\nScaling features...")

    scaler = RobustScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    )

    X_test = scaler.transform(
        test[FEATURES]
    )

    # =====================================
    # 5. TRAIN K-MEANS
    # =====================================

    print("\nTraining K-means model...")

    model = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=42,
        n_init=20
    )

    train["cluster"] = model.fit_predict(
        X_train
    )

    test["cluster"] = model.predict(
        X_test
    )

    print("\nModel training complete!")

    print("Clusters:", NUM_CLUSTERS)
    print("Inertia:", round(model.inertia_, 4))

    # =====================================
    # 6. ANALYZE TRAINING CLUSTERS
    # =====================================

    print("\nTRAINING CLUSTER RESULTS")
    print("========================")

    training_summary = train.groupby(
        "cluster"
    ).agg(
        observations=("close", "size"),
        average_return=("forward_return_pct", "mean"),
        average_volatility=("volatility_1h", "mean"),
        average_range=("range_pct", "mean"),
        average_volume_ratio=("log_volume_ratio", "mean")
    )

    print(training_summary.round(4).to_string())

    # Identify clusters with positive
    # average next-bar returns in training.
    favorable_clusters = training_summary[
        training_summary["average_return"] > 0
    ].index.tolist()

    print("\nClusters with positive average")
    print("training-period next-bar returns:")
    print(favorable_clusters)

    # =====================================
    # 7. TEST ON UNSEEN DATA
    # =====================================

    print("\nTESTING CLUSTER RESULTS")
    print("=======================")

    test_summary = test.groupby(
        "cluster"
    ).agg(
        observations=("close", "size"),
        average_return=("forward_return_pct", "mean"),
        average_volatility=("volatility_1h", "mean"),
        average_range=("range_pct", "mean")
    )

    print(test_summary.round(4).to_string())

    # Evaluate only clusters selected using
    # training data, without changing selection.
    selected_test = test[
        test["cluster"].isin(favorable_clusters)
    ]

    if len(selected_test) > 0:

        print("\nSELECTED CLUSTERS: TEST RESULTS")
        print("--------------------------------")

        print(
            "Observations:",
            len(selected_test)
        )

        print(
            "Average next-bar return (%):",
            round(
                selected_test["forward_return_pct"].mean(),
                5
            )
        )

    else:
        print("\nNo training-selected clusters appeared in test data.")

    # =====================================
    # 8. COMPARE HSY AND MCD
    # =====================================

    print("\nCLUSTER DISTRIBUTION BY STOCK")
    print("=============================")

    print(
        pd.crosstab(
            test["symbol"],
            test["cluster"]
        )
    )

    # =====================================
    # 9. SAVE RESULTS
    # =====================================

    results = pd.concat(
        [train, test]
    ).sort_values(
        ["symbol", "ts_utc"]
    )

    results["data_split"] = np.where(
        results["ts_utc"] < cutoff_date,
        "train",
        "test"
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved results:", OUTPUT_FILE)

    # =====================================
    # 10. VISUALIZE TEST CLUSTERS
    # =====================================

    plt.figure(figsize=(11, 7))

    for cluster in range(NUM_CLUSTERS):

        points = test[
            test["cluster"] == cluster
        ]

        plt.scatter(
            points["return_pct"],
            points["volatility_1h"],
            label=f"Cluster {cluster}",
            alpha=0.5,
            s=12
        )

    plt.xlabel("Five-Minute Price Return (%)")
    plt.ylabel("One-Hour Rolling Volatility (%)")

    plt.title(
        "K-Means Market Regimes: HSY and MCD (Test Data)"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=300
    )

    print("Saved graph:", GRAPH_FILE)

    plt.show()


if __name__ == "__main__":
    main()
