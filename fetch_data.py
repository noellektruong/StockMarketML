
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed


# Stocks for the project
STOCKS = ["HSY", "MCD"]

OUTPUT_FILE = "alpaca_hsy_mcd.csv"


def main():

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Missing Alpaca API credentials."
        )

    client = StockHistoricalDataClient(
        api_key,
        secret_key
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)

    request = StockBarsRequest(
        symbol_or_symbols=STOCKS,
        timeframe=TimeFrame(
            5,
            TimeFrameUnit.Minute
        ),
        start=start,
        end=end,
        feed=DataFeed.IEX
    )

    print("Downloading Alpaca historical data...")

    bars = client.get_stock_bars(request)

    data = bars.df.reset_index()

    if data.empty:
        raise ValueError(
            "Alpaca returned no stock data."
        )

    data = data.rename(
        columns={"timestamp": "ts_utc"}
    )

    data = data[
        [
            "symbol",
            "ts_utc",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    data = data.sort_values(
        ["symbol", "ts_utc"]
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDownload complete!")
    print("Rows downloaded:", len(data))
    print("\nRows by stock:")
    print(data["symbol"].value_counts())
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
