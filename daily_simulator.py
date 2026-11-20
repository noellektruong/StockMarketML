
import pandas as pd

from portfolio_state import (
    load_portfolio,
    save_portfolio,
    buy_virtual,
    sell_virtual,
)

DATA_FILE = "kmeans_clustered_data.csv"
POSITION_FRACTION = 0.10
STOCKS = ["HSY", "MCD"]

REQUIRED_COLUMNS = [
    "symbol",
    "ts_utc",
    "close",
    "individual_mood",
    "combined_mood",
]


def get_latest_data():
    print("\n[1] LOADING MARKET SIGNALS")

    data = pd.read_csv(DATA_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    data["ts_utc"] = pd.to_datetime(
        data["ts_utc"], utc=True, errors="coerce"
    )
    data["close"] = pd.to_numeric(data["close"], errors="coerce")

    data = data.dropna(subset=["ts_utc", "close"])
    data = data[data["symbol"].isin(STOCKS)]
    data = data.sort_values("ts_utc")

    if data.empty:
        raise ValueError("No valid HSY or MCD data found.")

    latest = data.groupby("symbol").tail(1).copy()

    if set(latest["symbol"]) != set(STOCKS):
        raise ValueError("Missing latest data for HSY or MCD.")

    latest["session_date"] = (
        latest["ts_utc"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    if latest["session_date"].nunique() != 1:
        raise ValueError("HSY and MCD data are from different sessions.")

    latest = latest.set_index("symbol")

    for symbol in STOCKS:
        if latest.loc[symbol, "close"] <= 0:
            raise ValueError(f"Invalid price for {symbol}.")

    print("[SUCCESS] Latest market data loaded.")

    return latest


def check_signal(row):
    individual = str(row["individual_mood"]).strip()
    combined = str(row["combined_mood"]).strip()

    individual_positive = individual.lower().startswith("positive-return")
    combined_positive = combined.lower().startswith("positive-return")

    reasons = []

    if not individual_positive:
        reasons.append("Individual mood is not positive-return")

    if not combined_positive:
        reasons.append("Combined mood is not positive-return")

    return individual_positive and combined_positive, reasons


def calculate_equity(portfolio, prices):
    stock_value = sum(
        float(position["shares"]) * prices[symbol]
        for symbol, position in portfolio["positions"].items()
        if symbol in prices
    )

    return float(portfolio["cash"]) + stock_value


def main():
    print("\n" + "=" * 55)
    print("DAILY VIRTUAL PORTFOLIO SIMULATOR")
    print("=" * 55)

    portfolio = load_portfolio()
    latest = get_latest_data()

    prices = latest["close"].astype(float).to_dict()
    session_date = latest["session_date"].iloc[0]

    print(f"\nTrading session: {session_date}")
    print(f"Starting cash: ${portfolio['cash']:,.2f}")

    if portfolio.get("last_processed_date") == session_date:
        print("\n[BLOCKED] Session already processed.")
        print("Duplicate protection prevented another run.")
        print("No new transactions were made.")
        return

    decisions = {}

    print("\n[2] MARKET SIGNAL DIAGNOSTICS")
    print("-" * 55)

    for symbol in STOCKS:
        row = latest.loc[symbol]
        positive, reasons = check_signal(row)

        decisions[symbol] = positive

        print(f"\n{symbol}")
        print(f"Price: ${prices[symbol]:.2f}")
        print(f"Individual mood: {row['individual_mood']}")
        print(f"Combined mood:   {row['combined_mood']}")

        if positive:
            print("[BUY CONDITION] Both signals are positive.")
        else:
            print("[NO BUY CONDITION]")
            for reason in reasons:
                print(f"  - {reason}")

    trade_count = 0

    print("\n[3] SELL DECISIONS")
    print("-" * 55)

    for symbol in STOCKS:
        if symbol not in portfolio["positions"]:
            print(f"[HOLD CASH] {symbol}: No existing position.")
            continue

        if decisions[symbol]:
            print(f"[HOLD] {symbol}: Existing position; signal remains positive.")
            continue

        shares = float(portfolio["positions"][symbol]["shares"])

        print(f"[SELL SIGNAL] {symbol}")
        sell_virtual(portfolio, symbol, shares, prices[symbol])
        trade_count += 1

    print("\n[4] BUY DECISIONS")
    print("-" * 55)

    for symbol in STOCKS:
        if not decisions[symbol]:
            print(f"[NO TRADE] {symbol}: Buy conditions not met.")
            continue

        if symbol in portfolio["positions"]:
            print(f"[HOLD] {symbol}: Already owned.")
            continue

        equity = calculate_equity(portfolio, prices)
        allocation = min(
            equity * POSITION_FRACTION,
            float(portfolio["cash"])
        )

        shares = allocation / prices[symbol] if prices[symbol] > 0 else 0

        if shares <= 0 or allocation <= 0:
            print(f"[NO TRADE] {symbol}: Insufficient virtual cash.")
            continue

        print(f"[BUY SIGNAL] {symbol}")
        print(f"Virtual allocation: ${allocation:,.2f}")
        print(f"Virtual shares: {shares:.6f}")

        buy_virtual(portfolio, symbol, shares, prices[symbol])
        trade_count += 1

    portfolio["last_processed_date"] = session_date
    save_portfolio(portfolio)

    total_equity = calculate_equity(portfolio, prices)
    stock_value = total_equity - float(portfolio["cash"])

    print("\n[5] FINAL VIRTUAL ACCOUNT")
    print("=" * 55)
    print(f"Cash:          ${portfolio['cash']:,.2f}")
    print(f"Stock value:   ${stock_value:,.2f}")
    print(f"Total equity:  ${total_equity:,.2f}")
    print(f"Change:        ${total_equity - 1_000_000:,.2f}")
    print(f"Positions:     {portfolio['positions']}")
    print(f"Transactions:  {len(portfolio['history'])}")

    print("\n[6] DAILY RESULT")

    if trade_count > 0:
        print(f"[SUCCESS] {trade_count} virtual transaction(s) completed.")
    else:
        print("[NO TRADES] No qualifying virtual transactions today.")

    print("=" * 55)


if __name__ == "__main__":
    main()
