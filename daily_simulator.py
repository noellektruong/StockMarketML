
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


def get_latest_data():
    data = pd.read_csv(DATA_FILE)

    data["ts_utc"] = pd.to_datetime(data["ts_utc"], utc=True)
    data = data.sort_values("ts_utc")

    latest = data.groupby("symbol").tail(1).copy()

    latest["session_date"] = (
        latest["ts_utc"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    if set(latest["symbol"]) != set(STOCKS):
        raise ValueError("Missing HSY or MCD data.")

    if latest["session_date"].nunique() != 1:
        raise ValueError("HSY and MCD latest data are from different sessions.")

    return latest.set_index("symbol")


def is_positive_signal(row):
    individual = str(row["individual_mood"]).lower()
    combined = str(row["combined_mood"]).lower()

    return (
        individual.startswith("positive-return")
        and combined.startswith("positive-return")
    )


def calculate_equity(portfolio, prices):
    stock_value = sum(
        position["shares"] * prices[symbol]
        for symbol, position in portfolio["positions"].items()
        if symbol in prices
    )
    return portfolio["cash"] + stock_value


def main():
    portfolio = load_portfolio()
    latest = get_latest_data()

    prices = latest["close"].astype(float).to_dict()
    session_date = latest["session_date"].iloc[0]

    print("\nDAILY VIRTUAL TRADING")
    print("=" * 45)
    print("Date:", session_date)

    if portfolio.get("last_processed_date") == session_date:
        print("Today's session has already been processed.")
        return

    # Sell existing positions when their signals are no longer positive.
    for symbol in STOCKS:
        row = latest.loc[symbol]
        positive = is_positive_signal(row)

        print(
            f"{symbol}: ${prices[symbol]:.2f} | "
            f"{row['individual_mood']} | {row['combined_mood']}"
        )

        if symbol in portfolio["positions"] and not positive:
            shares = portfolio["positions"][symbol]["shares"]
            print(f"SELL SIGNAL: {symbol}")
            sell_virtual(portfolio, symbol, shares, prices[symbol])

    # Buy stocks with positive individual AND combined moods.
    for symbol in STOCKS:
        row = latest.loc[symbol]

        if not is_positive_signal(row):
            print(f"{symbol}: No buy signal.")
            continue

        if symbol in portfolio["positions"]:
            print(f"{symbol}: Already held. Keeping position.")
            continue

        equity = calculate_equity(portfolio, prices)
        allocation = min(equity * POSITION_FRACTION, portfolio["cash"])

        if allocation <= 0:
            print(f"{symbol}: Insufficient virtual cash.")
            continue

        shares = allocation / prices[symbol]

        print(
            f"BUY SIGNAL: {symbol} | "
            f"Virtual allocation: ${allocation:,.2f}"
        )

        buy_virtual(portfolio, symbol, shares, prices[symbol])

    portfolio["last_processed_date"] = session_date
    save_portfolio(portfolio)

    total_equity = calculate_equity(portfolio, prices)

    print("\nFINAL VIRTUAL ACCOUNT")
    print("-" * 45)
    print(f"Cash: ${portfolio['cash']:,.2f}")
    print(f"Stock value: ${total_equity - portfolio['cash']:,.2f}")
    print(f"Total equity: ${total_equity:,.2f}")
    print(f"Positions: {portfolio['positions']}")
    print(f"Transactions: {len(portfolio['history'])}")
    print("=" * 45)


if __name__ == "__main__":
    main()
