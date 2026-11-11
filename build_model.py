
"""Prepare labeled stock examples and chronological train/test splits.
No model training is performed in this script.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

INPUT_FILE = Path("hsy_mcd_2025.csv")
LABELED_FILE = Path("labeled_stock_data.csv")
TRAIN_FILE = Path("stock_train.csv")
TEST_FILE = Path("stock_test.csv")

TRAIN_END = pd.Timestamp("2025-10-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")


def main():
    # Read the existing CSV.
    data = pd.read_csv(INPUT_FILE)

    # Convert timestamps and sort each stock chronologically.
    data["ts_utc"] = pd.to_datetime(data["ts_utc"], utc=True)
    data = data.sort_values(["symbol", "ts_utc"]).copy()

    # Create the input feature: difference between high and low.
    data["price_range"] = data["high"] - data["low"]

    # Label: closing price exactly 15 minutes in the future.
    grouped = data.groupby("symbol")
    data["future_close"] = grouped["close"].shift(-3)
    data["future_timestamp"] = grouped["ts_utc"].shift(-3)

    # Keep only observations exactly 15 minutes apart.
    data = data[
        data["future_timestamp"]
        == data["ts_utc"] + pd.Timedelta(minutes=15)
    ].copy()

    # Keep the columns needed for modeling.
    labeled = data[
        [
            "symbol",
            "ts_utc",
            "future_timestamp",
            "close",
            "volume",
            "price_range",
            "future_close",
        ]
    ].dropna().copy()

    # Chronological training split: January through September.
    train = labeled[
        (labeled["ts_utc"] < TRAIN_END)
        & (labeled["future_timestamp"] < TRAIN_END)
    ].copy()

    # Chronological testing split: October through December.
    test = labeled[
        (labeled["ts_utc"] >= TRAIN_END)
        & (labeled["future_timestamp"] < TEST_END)
    ].copy()

    # Save separate local CSV files.
    labeled.to_csv(LABELED_FILE, index=False)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    # Create separate model objects, but DO NOT train them today.
    models = {
        "HSY": LinearRegression(),
        "MCD": LinearRegression(),
    }

    print("=== DATA PREPARATION COMPLETE ===")
    print("Labeled examples:", len(labeled))
    print("Training examples:", len(train))
    print("Testing examples:", len(test))
    print("\nExamples by stock:")

    for symbol in ("HSY", "MCD"):
        print(
            symbol,
            "| Train:", len(train[train["symbol"] == symbol]),
            "| Test:", len(test[test["symbol"] == symbol]),
        )

    print("\nCreated files:")
    print(LABELED_FILE)
    print(TRAIN_FILE)
    print(TEST_FILE)
    print("\nModels initialized:", list(models.keys()))
    print("No model training performed.")


if __name__ == "__main__":
    main()
