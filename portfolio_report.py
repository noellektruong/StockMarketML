
import pandas as pd
from portfolio_state import load_portfolio

DATA_FILE = "alpaca_hsy_mcd.csv"
STARTING_CASH = 1_000_000.00


def get_latest_prices():
    data = pd.read_csv(DATA_FILE)

    data["ts_utc"] = pd.to_datetime(data["ts_utc"], utc=True)
    data = data.sort_values("ts_utc")

    latest = data.groupby("symbol").tail(1)

    return {
        row["symbol"]: float(row["close"])
        for _, row in latest.iterrows()
    }


def show_report():
    portfolio = load_portfolio()
    prices = get_latest_prices()

    cash = portfolio["cash"]
    stock_value = 0
    total_cost = 0

    print("\nALPACA DATA - VIRTUAL PORTFOLIO")
    print("=" * 48)

    print(f"Available virtual cash: ${cash:,.2f}")

    print("\nLATEST MARKET PRICES")

    for symbol, price in prices.items():
        print(f"{symbol}: ${price:.2f}")

    print("\nOPEN POSITIONS")

    if not portfolio["positions"]:
        print("No stocks currently held.")

    for symbol, position in portfolio["positions"].items():
        shares = position["shares"]
        average_price = position["average_price"]

        if symbol not in prices:
            print(f"{symbol}: Current price unavailable")
            continue

        current_price = prices[symbol]
        market_value = shares * current_price
        cost_basis = shares * average_price
        profit = market_value - cost_basis

        stock_value += market_value
        total_cost += cost_basis

        print(f"\n{symbol}")
        print(f"Shares: {shares}")
        print(f"Average purchase price: ${average_price:.2f}")
        print(f"Current price: ${current_price:.2f}")
        print(f"Market value: ${market_value:,.2f}")
        print(f"Unrealized P/L: ${profit:,.2f}")

    total_value = cash + stock_value
    change = total_value - STARTING_CASH

    print("\n" + "=" * 48)
    print(f"Cash: ${cash:,.2f}")
    print(f"Stock value: ${stock_value:,.2f}")
    print(f"TOTAL PORTFOLIO: ${total_value:,.2f}")
    print(f"Total change: ${change:,.2f}")
    print("=" * 48)


if __name__ == "__main__":
    show_report()
