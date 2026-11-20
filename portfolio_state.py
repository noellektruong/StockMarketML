
import json
from pathlib import Path
from datetime import datetime, timezone

STATE_FILE = Path("portfolio_state.json")
STARTING_CASH = 1_000_000.00


def load_portfolio():
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)

    portfolio = {
        "cash": STARTING_CASH,
        "positions": {},
        "history": [],
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    save_portfolio(portfolio)
    return portfolio


def save_portfolio(portfolio):
    temporary_file = STATE_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(portfolio, file, indent=4)

    temporary_file.replace(STATE_FILE)


def buy_virtual(portfolio, symbol, shares, price):
    cost = shares * price

    if shares <= 0 or price <= 0:
        raise ValueError("Shares and price must be positive.")

    if cost > portfolio["cash"]:
        print("Not enough virtual cash.")
        return portfolio

    position = portfolio["positions"].setdefault(
        symbol, {"shares": 0, "average_price": 0}
    )

    old_shares = position["shares"]
    new_shares = old_shares + shares

    position["average_price"] = (
        (old_shares * position["average_price"] + cost) / new_shares
    )
    position["shares"] = new_shares
    portfolio["cash"] -= cost

    portfolio["history"].append({
        "time": datetime.now(timezone.utc).isoformat(),
        "action": "BUY",
        "symbol": symbol,
        "shares": shares,
        "price": price,
        "total": cost
    })

    save_portfolio(portfolio)
    return portfolio


def sell_virtual(portfolio, symbol, shares, price):
    if shares <= 0 or price <= 0:
        raise ValueError("Shares and price must be positive.")

    position = portfolio["positions"].get(symbol)

    if not position or shares > position["shares"]:
        print("Not enough virtual shares to sell.")
        return portfolio

    proceeds = shares * price
    position["shares"] -= shares
    portfolio["cash"] += proceeds

    portfolio["history"].append({
        "time": datetime.now(timezone.utc).isoformat(),
        "action": "SELL",
        "symbol": symbol,
        "shares": shares,
        "price": price,
        "total": proceeds
    })

    if position["shares"] == 0:
        del portfolio["positions"][symbol]

    save_portfolio(portfolio)
    return portfolio


if __name__ == "__main__":
    account = load_portfolio()

    print("VIRTUAL PORTFOLIO")
    print("-----------------")
    print(f"Cash: ${account['cash']:,.2f}")
    print(f"Positions: {account['positions']}")
    print(f"Recorded transactions: {len(account['history'])}")
