"""Paper-trading backtest for the strict 50/15/15/20 K-means pipeline.

Historical simulation only. No broker connection, API credentials, or orders.
The K-means scaler/centroids must be trained on the first 50% block only;
regime selection uses discovery + validation; final evaluation uses the last
20% block once.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

INPUT_FILE = Path("kmeans_clustered_data.csv")
TRADES_FILE = Path("paper_trading_trades.csv")
REGIME_FILE = Path("paper_trading_regime_stats.csv")
SUMMARY_FILE = Path("paper_trading_summary.csv")
EQUITY_FILE = Path("paper_trading_equity.csv")

STOCKS = ["HSY", "MCD"]
BAR_MINUTES = 5
HOLD_BARS = 6
HORIZON_OFFSET = HOLD_BARS + 1  # signal row -> next-open entry -> 6 intervals -> exit open
MIN_OBSERVATIONS_PER_SEGMENT = 50
MIN_WIN_RATE_PCT = 52.0
COST_BPS_PER_SIDE = 5.0
POSITION_FRACTION = 0.10
INITIAL_CAPITAL_PER_STOCK = 10_000.0
ROUND_TRIP_COST_BPS = COST_BPS_PER_SIDE * 2.0
COST_RATE = COST_BPS_PER_SIDE / 10_000.0
HOLDING_MINUTES = HOLD_BARS * BAR_MINUTES

REQUIRED_COLUMNS = {
    "symbol", "ts_utc", "open", "close", "individual_cluster",
    "combined_cluster", "data_split", "combined_data_split",
    "individual_mood", "combined_mood",
}
TRADE_COLUMNS = [
    "symbol", "trade_number", "signal_timestamp_utc", "entry_timestamp_utc",
    "exit_timestamp_utc", "individual_cluster", "combined_cluster",
    "individual_mood", "combined_mood", "discovery_observations",
    "discovery_mean_return_bps", "validation_observations",
    "validation_mean_return_bps", "validation_win_rate_pct", "entry_price",
    "exit_price", "holding_bars", "gross_asset_return_pct",
    "net_asset_return_pct", "entry_fee_dollars", "exit_fee_dollars",
    "position_fraction", "account_equity_before", "account_equity_after",
    "account_return_pct",
]


def load_clustered_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE}. Run kmeans_market_model.py first.")
    data = pd.read_csv(INPUT_FILE)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError("Clustered CSV is missing columns: " + ", ".join(sorted(missing)))

    data["ts_utc"] = pd.to_datetime(data["ts_utc"], utc=True, errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    for col in ["data_split", "combined_data_split"]:
        data[col] = data[col].astype(str).str.lower().str.strip()
    for col in ["open", "close", "individual_cluster", "combined_cluster"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["individual_mood", "combined_mood"]:
        data[col] = data[col].fillna("").astype(str)

    data = data[data["symbol"].isin(STOCKS)].copy()
    data = data.dropna(subset=["ts_utc", "open", "close", "individual_cluster", "combined_cluster"])
    data = data[(data["open"] > 0) & (data["close"] > 0)].copy()
    allowed = {"fit", "discovery", "validation", "test"}
    data = data[
        data["data_split"].isin(allowed)
        & (data["data_split"] == data["combined_data_split"])
    ].copy()
    data["individual_cluster"] = data["individual_cluster"].astype(int)
    data["combined_cluster"] = data["combined_cluster"].astype(int)
    data["session_date_et"] = data["ts_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    data = data.sort_values(["symbol", "ts_utc"]).reset_index(drop=True)

    if set(STOCKS).difference(data["symbol"].unique()):
        raise ValueError("Both HSY and MCD must be present in clustered data.")
    for symbol, stock in data.groupby("symbol"):
        if stock["ts_utc"].duplicated().any():
            raise ValueError(f"Duplicate timestamps found for {symbol}.")

    present_periods = set(data["data_split"].unique())
    if present_periods != allowed:
        raise ValueError(f"Expected all four periods {sorted(allowed)}; found {sorted(present_periods)}. Run the updated kmeans_market_model.py first.")
    return data


def add_forward_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    grouped = result.groupby("symbol", sort=False)
    result["entry_open_next_bar"] = grouped["open"].shift(-1)
    result["entry_timestamp_next_bar"] = grouped["ts_utc"].shift(-1)
    result["entry_period_next_bar"] = grouped["data_split"].shift(-1)
    result["exit_open_horizon"] = grouped["open"].shift(-HORIZON_OFFSET)
    result["exit_timestamp_horizon"] = grouped["ts_utc"].shift(-HORIZON_OFFSET)
    result["exit_period_horizon"] = grouped["data_split"].shift(-HORIZON_OFFSET)
    result["horizon_return_bps"] = (
        result["exit_open_horizon"] / result["entry_open_next_bar"] - 1.0
    ) * 10_000.0
    result["horizon_elapsed_minutes"] = (
        result["exit_timestamp_horizon"] - result["entry_timestamp_next_bar"]
    ).dt.total_seconds() / 60.0
    result["exit_session_date_et"] = (
        result["exit_timestamp_horizon"].dt.tz_convert("America/New_York").dt.date.astype("string")
    )

    consecutive = pd.Series(True, index=result.index)
    for step in range(1, HORIZON_OFFSET + 1):
        future_ts = grouped["ts_utc"].shift(-step)
        expected_ts = result["ts_utc"] + pd.Timedelta(minutes=BAR_MINUTES * step)
        consecutive &= future_ts.eq(expected_ts)

    same_selection_segment = (
        result["data_split"].isin(["discovery", "validation"])
        & (result["data_split"] == result["entry_period_next_bar"])
        & (result["data_split"] == result["exit_period_horizon"])
    )
    valid = (
        same_selection_segment
        & result["horizon_return_bps"].notna()
        & np.isfinite(result["horizon_return_bps"])
        & (result["horizon_elapsed_minutes"] == HOLDING_MINUTES)
        & (result["session_date_et"] == result["exit_session_date_et"])
        & consecutive
    )
    result["valid_selection_outcome"] = valid
    print("\nChronological blocks found:")
    for period in ["fit", "discovery", "validation", "test"]:
        rows = result[result["data_split"] == period]
        print(f"{period.title():<12} {len(rows):>7,} rows | {rows['ts_utc'].min()} through {rows['ts_utc'].max()}")
    return result


def evaluate_regimes(data: pd.DataFrame):
    selection_rows = data[data["valid_selection_outcome"]].copy()
    if selection_rows.empty:
        raise ValueError("No complete discovery/validation forward outcomes are available.")

    keys = ["symbol", "individual_cluster", "combined_cluster"]
    stats = selection_rows.groupby(keys + ["data_split"], as_index=False).agg(
        observations=("horizon_return_bps", "size"),
        mean_return_bps=("horizon_return_bps", "mean"),
        median_return_bps=("horizon_return_bps", "median"),
        win_rate_pct=("horizon_return_bps", lambda values: (values > 0).mean() * 100.0),
    )
    wide = stats.pivot(index=keys, columns="data_split")
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()

    all_pairs = pd.MultiIndex.from_product(
        [STOCKS, range(3), range(3)], names=keys
    ).to_frame(index=False)
    wide = all_pairs.merge(wide, on=keys, how="left")
    for period in ["discovery", "validation"]:
        for metric in ["observations", "mean_return_bps", "median_return_bps", "win_rate_pct"]:
            col = f"{metric}_{period}"
            if col not in wide:
                wide[col] = np.nan

    wide["estimated_edge_bps_discovery"] = wide["mean_return_bps_discovery"] - ROUND_TRIP_COST_BPS
    wide["estimated_edge_bps_validation"] = wide["mean_return_bps_validation"] - ROUND_TRIP_COST_BPS
    enough = (
        (wide["observations_discovery"] >= MIN_OBSERVATIONS_PER_SEGMENT)
        & (wide["observations_validation"] >= MIN_OBSERVATIONS_PER_SEGMENT)
    )
    cost_covered = (
        (wide["mean_return_bps_discovery"] > ROUND_TRIP_COST_BPS)
        & (wide["mean_return_bps_validation"] > ROUND_TRIP_COST_BPS)
    )
    positive_medians = (
        (wide["median_return_bps_discovery"] > 0)
        & (wide["median_return_bps_validation"] > 0)
    )
    win_rate_pass = (
        (wide["win_rate_pct_discovery"] >= MIN_WIN_RATE_PCT)
        & (wide["win_rate_pct_validation"] >= MIN_WIN_RATE_PCT)
    )
    wide["selected_for_test"] = enough & cost_covered & positive_medians & win_rate_pass

    def reason(row):
        if pd.isna(row["observations_discovery"]) or row["observations_discovery"] < MIN_OBSERVATIONS_PER_SEGMENT:
            return "Fail: too few discovery observations"
        if pd.isna(row["observations_validation"]) or row["observations_validation"] < MIN_OBSERVATIONS_PER_SEGMENT:
            return "Fail: too few validation observations"
        if row["mean_return_bps_discovery"] <= ROUND_TRIP_COST_BPS or row["mean_return_bps_validation"] <= ROUND_TRIP_COST_BPS:
            return "Fail: mean return does not exceed estimated costs in both periods"
        if row["median_return_bps_discovery"] <= 0 or row["median_return_bps_validation"] <= 0:
            return "Fail: median return is not positive in both periods"
        if row["win_rate_pct_discovery"] < MIN_WIN_RATE_PCT or row["win_rate_pct_validation"] < MIN_WIN_RATE_PCT:
            return "Fail: win rate is below threshold in one or both periods"
        return "Selected: passed both training segments"

    wide["selection_reason"] = wide.apply(reason, axis=1)
    wide = wide.sort_values(
        ["symbol", "selected_for_test", "estimated_edge_bps_validation"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    selected = {symbol: set() for symbol in STOCKS}
    for row in wide[wide["selected_for_test"]].itertuples():
        selected[row.symbol].add((int(row.individual_cluster), int(row.combined_cluster)))
    return wide, selected


def valid_test_window(stock: pd.DataFrame, signal_index: int, exit_index: int) -> bool:
    if exit_index >= len(stock):
        return False
    window = stock.iloc[signal_index:exit_index + 1]
    if len(window) != HORIZON_OFFSET + 1:
        return False
    if not (window["data_split"] == "test").all():
        return False
    if not (window["combined_data_split"] == "test").all():
        return False
    if window["session_date_et"].nunique() != 1:
        return False
    return bool((window["ts_utc"].diff().dropna() == pd.Timedelta(minutes=BAR_MINUTES)).all())


def simulate_stock(symbol: str, test_stock: pd.DataFrame, selected_pairs: set, regime_stats: pd.DataFrame):
    stock = test_stock.sort_values("ts_utc").reset_index(drop=True).copy()
    stats = regime_stats[regime_stats["symbol"] == symbol].set_index(
        ["individual_cluster", "combined_cluster"]
    )
    equity = INITIAL_CAPITAL_PER_STOCK
    peak_equity = equity
    max_drawdown_pct = 0.0
    trades = []
    equity_rows = [
        {"symbol": symbol, "timestamp_utc": row.ts_utc, "account_equity": np.nan, "in_position": False}
        for row in stock.itertuples()
    ]
    current_index = 0
    trade_number = 0
    held_intervals = 0

    while current_index < len(stock):
        signal = stock.iloc[current_index]
        pair = (int(signal["individual_cluster"]), int(signal["combined_cluster"]))
        if pair not in selected_pairs:
            current_index += 1
            continue
        entry_index = current_index + 1
        exit_index = current_index + HORIZON_OFFSET
        if not valid_test_window(stock, current_index, exit_index):
            current_index += 1
            continue

        entry = stock.iloc[entry_index]
        exit_row = stock.iloc[exit_index]
        entry_price = float(entry["open"])
        exit_price = float(exit_row["open"])
        if entry_price <= 0 or exit_price <= 0:
            current_index += 1
            continue

        regime = stats.loc[pair]
        equity_before = equity
        position_notional = equity_before * POSITION_FRACTION
        entry_fee = position_notional * COST_RATE
        shares = position_notional / entry_price
        cash_after_entry = equity_before - position_notional - entry_fee
        exit_notional = shares * exit_price
        exit_fee = exit_notional * COST_RATE

        for mark_index in range(entry_index, exit_index + 1):
            mark_price = float(stock.iloc[mark_index]["open"])
            marked_equity = cash_after_entry + shares * mark_price
            if mark_index == exit_index:
                marked_equity -= exit_fee
            equity_rows[mark_index]["account_equity"] = marked_equity
            equity_rows[mark_index]["in_position"] = True
            peak_equity = max(peak_equity, marked_equity)
            max_drawdown_pct = min(max_drawdown_pct, (marked_equity / peak_equity - 1.0) * 100.0)

        equity_after = cash_after_entry + exit_notional - exit_fee
        gross_return_pct = (exit_price / entry_price - 1.0) * 100.0
        net_return_pct = ((exit_price / entry_price) * (1.0 - COST_RATE) - (1.0 + COST_RATE)) * 100.0
        account_return_pct = (equity_after / equity_before - 1.0) * 100.0
        trade_number += 1
        trades.append({
            "symbol": symbol,
            "trade_number": trade_number,
            "signal_timestamp_utc": signal["ts_utc"].isoformat(),
            "entry_timestamp_utc": entry["ts_utc"].isoformat(),
            "exit_timestamp_utc": exit_row["ts_utc"].isoformat(),
            "individual_cluster": pair[0],
            "combined_cluster": pair[1],
            "individual_mood": signal["individual_mood"],
            "combined_mood": signal["combined_mood"],
            "discovery_observations": int(regime["observations_discovery"]),
            "discovery_mean_return_bps": float(regime["mean_return_bps_discovery"]),
            "validation_observations": int(regime["observations_validation"]),
            "validation_mean_return_bps": float(regime["mean_return_bps_validation"]),
            "validation_win_rate_pct": float(regime["win_rate_pct_validation"]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "holding_bars": HOLD_BARS,
            "gross_asset_return_pct": gross_return_pct,
            "net_asset_return_pct": net_return_pct,
            "entry_fee_dollars": entry_fee,
            "exit_fee_dollars": exit_fee,
            "position_fraction": POSITION_FRACTION,
            "account_equity_before": equity_before,
            "account_equity_after": equity_after,
            "account_return_pct": account_return_pct,
        })
        equity = equity_after
        held_intervals += HOLD_BARS
        current_index = exit_index

    last_equity = INITIAL_CAPITAL_PER_STOCK
    for row in equity_rows:
        if pd.isna(row["account_equity"]):
            row["account_equity"] = last_equity
        else:
            last_equity = float(row["account_equity"])
    equity_frame = pd.DataFrame(equity_rows)
    equity_frame["equity_peak"] = equity_frame["account_equity"].cummax()
    equity_frame["drawdown_pct"] = (equity_frame["account_equity"] / equity_frame["equity_peak"] - 1.0) * 100.0
    if not equity_frame.empty:
        max_drawdown_pct = min(max_drawdown_pct, float(equity_frame["drawdown_pct"].min()))

    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        trade_frame = pd.DataFrame(columns=TRADE_COLUMNS)
    summary = {
        "symbol": symbol,
        "test_observations": len(stock),
        "selected_regime_combinations": len(selected_pairs),
        "trade_count": len(trade_frame),
        "win_rate_pct": float((trade_frame["net_asset_return_pct"] > 0).mean() * 100.0) if not trade_frame.empty else 0.0,
        "time_in_market_pct": held_intervals / max(len(stock), 1) * 100.0,
        "strategy_total_return_pct": (equity / INITIAL_CAPITAL_PER_STOCK - 1.0) * 100.0,
        "strategy_final_equity": equity,
        "strategy_max_drawdown_pct": max_drawdown_pct,
        "initial_capital": INITIAL_CAPITAL_PER_STOCK,
        "position_fraction_pct": POSITION_FRACTION * 100.0,
        "assumed_cost_bps_per_side": COST_BPS_PER_SIDE,
        "holding_horizon_minutes": HOLDING_MINUTES,
    }
    return trade_frame, equity_frame, summary


def add_buy_hold_benchmark(summary: dict, test_stock: pd.DataFrame):
    stock = test_stock.sort_values("ts_utc").reset_index(drop=True)
    if stock.empty:
        summary["buy_hold_portfolio_return_pct"] = np.nan
        return summary
    first_open = float(stock.iloc[0]["open"])
    final_close = float(stock.iloc[-1]["close"])
    notional = INITIAL_CAPITAL_PER_STOCK * POSITION_FRACTION
    shares = notional / first_open
    cash = INITIAL_CAPITAL_PER_STOCK - notional - notional * COST_RATE
    final_notional = shares * final_close
    final_equity = cash + final_notional - final_notional * COST_RATE
    summary["buy_hold_asset_return_pct"] = (final_close / first_open - 1.0) * 100.0
    summary["buy_hold_portfolio_return_pct"] = (final_equity / INITIAL_CAPITAL_PER_STOCK - 1.0) * 100.0
    summary["buy_hold_final_equity"] = final_equity
    return summary


def main() -> int:
    print("\nK-MEANS STRICT 50/15/15/20 PAPER BACKTEST")
    print("==========================================")
    print("Historical simulation only. No broker connection or orders.")
    print("Timeline: 50% model fit | 15% discovery | 15% validation | 20% untouched test")
    print(f"Holding horizon: {HOLD_BARS} bars ({HOLDING_MINUTES} minutes)")
    print("Entry: next bar open | Exit: six intervals later at bar open")
    print(f"Minimum observations in EACH selection segment: {MIN_OBSERVATIONS_PER_SEGMENT}")
    print(f"Minimum win rate in EACH selection segment: {MIN_WIN_RATE_PCT:.1f}%")
    print(f"Estimated round-trip cost: {ROUND_TRIP_COST_BPS:.2f} bps")
    print(f"Position allocation: {POSITION_FRACTION:.0%} of account equity")

    try:
        data = add_forward_outcomes(load_clustered_data())
        regime_stats, selected_pairs = evaluate_regimes(data)
        regime_stats.to_csv(REGIME_FILE, index=False)

        print("\nDISCOVERY / VALIDATION REGIME RESULTS")
        print("=====================================")
        columns = [
            "symbol", "individual_cluster", "combined_cluster",
            "observations_discovery", "mean_return_bps_discovery", "win_rate_pct_discovery",
            "observations_validation", "mean_return_bps_validation", "win_rate_pct_validation",
            "selected_for_test", "selection_reason",
        ]
        print(regime_stats[columns].round(2).to_string(index=False))

        print("\nREGIMES SELECTED FOR UNTOUCHED TEST")
        print("===================================")
        for symbol in STOCKS:
            chosen = sorted(selected_pairs[symbol])
            print(f"{symbol}: {chosen if chosen else 'None — no pair passed both selection periods'}")

        trade_frames, equity_frames, summaries = [], [], []
        for symbol in STOCKS:
            test_stock = data[
                (data["symbol"] == symbol)
                & (data["data_split"] == "test")
                & (data["combined_data_split"] == "test")
            ].copy()
            trades, equity, summary = simulate_stock(symbol, test_stock, selected_pairs[symbol], regime_stats)
            summary = add_buy_hold_benchmark(summary, test_stock)
            trade_frames.append(trades)
            equity_frames.append(equity)
            summaries.append(summary)

        trades_output = pd.concat(trade_frames, ignore_index=True)
        equity_output = pd.concat(equity_frames, ignore_index=True)
        summary_output = pd.DataFrame(summaries)
        trades_output.to_csv(TRADES_FILE, index=False)
        equity_output.to_csv(EQUITY_FILE, index=False)
        summary_output.to_csv(SUMMARY_FILE, index=False)

        print("\nUNTOUCHED TEST RESULTS")
        print("======================")
        display = [
            "symbol", "test_observations", "selected_regime_combinations", "trade_count",
            "win_rate_pct", "time_in_market_pct", "strategy_total_return_pct",
            "buy_hold_portfolio_return_pct", "strategy_max_drawdown_pct",
        ]
        print(summary_output[display].round(4).to_string(index=False))
        if trades_output.empty:
            print("\nNo test trades. No regime combination passed both discovery and validation.")
        else:
            print("\nSIMULATED TRADE SAMPLE")
            sample = ["symbol", "signal_timestamp_utc", "entry_price", "exit_price", "net_asset_return_pct", "account_return_pct"]
            print(trades_output[sample].head(10).round(4).to_string(index=False))

        print(f"\nSaved regime statistics: {REGIME_FILE}")
        print(f"Saved simulated trades: {TRADES_FILE}")
        print(f"Saved equity curve: {EQUITY_FILE}")
        print(f"Saved summary: {SUMMARY_FILE}")
        print("No Alpaca trading endpoint was called; no orders were placed.")
        return 0
    except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
