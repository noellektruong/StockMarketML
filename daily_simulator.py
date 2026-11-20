
import math
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

TRADE_COUNT = 0


def get_latest_data():
    print("\n[DEBUG] Loading K-means data...")

    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"ERROR: {DATA_FILE} does not exist. "
            "K-means analysis may not have generated the file."
        )

    required_columns = [
        "symbol",
        "ts_utc",
        "close",
        "individual_mood",
        "combined_mood",
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["ts_utc"] = pd.to_datetime(
        data["ts_utc"], utc=True, errors="coerce"
    )

    data = data.dropna(subset=["ts_utc"])
    data = data.sort_values("ts_utc")

    latest = data.groupby("symbol").tail(1).copy()

    latest["session_date"] = (
        latest["ts_utc"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    missing_stocks = set(STOCKS) - set(latest["symbol"])

    if missing_stocks:
        raise ValueError(
            f"Missing latest market data for: {sorted(missing_stocks)}"
        )

    latest = latest[latest["symbol"].isin(STOCKS)]

    if latest["session_date"].nunique() != 1:
        raise ValueError(
            "HSY and MCD have data from different trading sessions."
        )

    for symbol, row in latest.set_index("symbol").iterrows():
        price = pd.to_numeric(row["close"], errors="coerce")

        if pd.isna(price) or not math.isfinite(float(price)) or price <= 0:
            raise ValueError(
                f"{symbol} has an invalid latest closing price: "
                f"{row['close']}"
            )

    print("[DEBUG] Market data loaded successfully.")

    return latest.set_index("symbol")


def check_signal(row):
    individual = str(row["individual_mood"]).strip().lower()
    combined = str(row["combined_mood"]).strip().lower()

    reasons = []

    if not individual.startswith("positive-return"):
        reasons.append(
            f"Individual mood is '{row['individual_mood']}', "
            "not Positive-return."
        )

    if not combined.startswith("positive-return"):
        reasons.append(
            f"Combined mood is '{row['combined_mood']}', "
            "not Positive-return."
        )

    if reasons:
        return False, reasons

    return True, ["Both individual and combined moods are positive."]


def calculate_equity(portfolio, prices):
    stock_value = sum(
        position["shares"] * prices[symbol]
        for symbol, position in portfolio["positions"].items()
        if symbol in prices
    )

    return portfolio["cash"] + stock_value


def main():
    global TRADE_COUNT

    print("\n" + "=" * 55)
    print("DAILY VIRTUAL TRADING - DEBUG MODE")
    print("=" * 55)

    portfolio = load_portfolio()
    latest = get_latest_data()

    prices = latest["close"].astype(float).to_dict()
    session_date = latest["session_date"].iloc[0]

    print(f"\nTrading session: {session_date}")
    print(f"Starting cash: ${portfolio['cash']:,.2f}")

    # Prevent processing the same trading session twice.
    if portfolio.get("last_processed_date") == session_date:
        print("\n[BLOCKED] This session was already processed.")
        print("Reason: Duplicate-trading protection is active.")

        for symbol in STOCKS:
            print(f"{symbol}: No action because today's session was already processed.")

        return

    signals = {}

    print("\n" + "-" * 55)
    print("MARKET SIGNAL DIAGNOSTICS")
    print("-" * 55)

    for symbol in STOCKS:
        row = latest.loc[symbol]
        positive, reasons = check_signal(row)

        signals[symbol] = positive

        print(f"\n{symbol}")
        print(f"  Latest price: ${prices[symbol]:.2f}")
        print(f"  Individual mood: {row['individual_mood']}")
        print(f"  Combined mood: {row['combined_mood']}")

        if positive:
            print("  [PASS] Both signal conditions passed.")
        else:
            print("  [FAIL] Buy conditions not met:")
            for reason in reasons:
                print(f"    - {reason}")

    # SELL existing positions if their signals are no longer positive.
    print("\n" + "-" * 55)
    print("SELL DECISIONS")
    print("-" * 55)

    for symbol in STOCKS:
        if symbol not in portfolio["positions"]:
            print(f"{symbol}: No sell needed. No existing position.")
            continue

        if signals[symbol]:
            print(
                f"{symbol}: HOLD. Existing position retained because "
                "both signals remain positive."
            )
            continue

        shares = portfolio["positions"][symbol]["shares"]

        print(
            f"{symbol}: SELL. Signal conditions are no longer positive."
        )

        try:
            sell_virtual(portfolio, symbol, shares, prices[symbol])
            TRADE_COUNT += 1
            print(f"  [SUCCESS] Sold {shares:.6f} shares.")
        except Exception as error:
            print(f"  [ERROR] Sell failed: {error}")
            raise

    # BUY stocks when both moods are positive.
    print("\n" + "-" * 55)
    print("BUY DECISIONS")
    print("-" * 55)

    for symbol in STOCKS:
        print(f"\n{symbol}:")

        if not signals[symbol]:
            print("[NO BUY] Signal requirements failed.")

            _, reasons = check_signal(latest.loc[symbol])

            for reason in reasons:
                print(f"  Reason: {reason}")

            continue

        if symbol in portfolio["positions"]:
            print("[NO BUY] Already holding this stock.")
            print("  Reason: Duplicate positions are not opened.")
            continue

        equity = calculate_equity(portfolio, prices)

        allocation = min(
            equity * POSITION_FRACTION,
            portfolio["cash"]
        )

        if allocation <= 0:
            print("[NO BUY] Insufficient virtual cash.")
            print(f"  Available cash: ${portfolio['cash']:,.2f}")
            continue

        shares = allocation / prices[symbol]

        if not math.isfinite(shares) or shares <= 0:
            print("[NO BUY] Calculated share quantity is invalid.")
            continue

        print("[BUY SIGNAL] Both moods are positive.")
        print(f"  Allocation: ${allocation:,.2f}")
        print(f"  Price: ${prices[symbol]:.2f}")
        print(f"  Shares: {shares:.6f}")

        try:
            buy_virtual(portfolio, symbol, shares, prices[symbol])
            TRADE_COUNT += 1
            print("[SUCCESS] Virtual purchase recorded.")
        except Exception as error:
            print(f"[ERROR] Buy failed: {error}")
            raise

    # Save the date even if no trades occurred.
    portfolio["last_processed_date"] = session_date
    save_portfolio(portfolio)

    total_equity = calculate_equity(portfolio, prices)
    stock_value = total_equity - portfolio["cash"]

    print("\n" + "=" * 55)
    print("FINAL VIRTUAL ACCOUNT")
    print("=" * 55)

    print(f"Cash: ${portfolio['cash']:,.2f}")
    print(f"Stock value: ${stock_value:,.2f}")
    print(f"Total equity: ${total_equity:,.2f}")
    print(f"Positions: {portfolio['positions']}")
    print(f"Recorded transactions: {len(portfolio['history'])}")

    print("\n" + "-" * 55)
    print("DAILY TRADING DIAGNOSTICS")
    print("-" * 55)

    if TRADE_COUNT == 0:
        print("[NO TRADES] No virtual transactions occurred today.")
        print("See the BUY and SELL diagnostics above for reasons.")
    else:
        print(f"[SUCCESS] {TRADE_COUNT} virtual transaction(s) recorded.")

    print("=" * 55)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n[FATAL ERROR] Daily simulator could not complete.")
        print(f"Reason: {type(error).__name__}: {error}")
        raise
