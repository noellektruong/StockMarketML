
"""IB Computer Science SL2 — Export stock data to CSV."""

import sqlite3
import csv
from pathlib import Path

# Original classroom database (READ ONLY)
DB_PATH = Path(r"\\192.168.1.95\CS-CLASSROOM\intraday_5m.sqlite")

DB_URI = (
    "file:////192.168.1.95/CS-CLASSROOM/"
    "intraday_5m.sqlite?mode=ro&immutable=1"
)

STOCKS_TO_EXPLORE = ("HSY", "MCD")

START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

OUTPUT_FILE = Path("hsy_mcd_2025.csv")


def main():

    if not DB_PATH.is_file():
        raise FileNotFoundError(
            "Cannot access the classroom NAS database. "
            "Connect using the CS-Students account."
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. Not overwriting it."
        )

    # Connect to the original database in READ-ONLY mode.
    with sqlite3.connect(DB_URI, uri=True) as conn:

        conn.execute("PRAGMA query_only = ON")

        cursor = conn.execute("""
            SELECT
                symbol,
                ts_utc,
                open,
                high,
                low,
                close,
                volume
            FROM bars_5m
            WHERE symbol IN (?, ?)
              AND ts_utc >= ?
              AND ts_utc < ?
            ORDER BY symbol, ts_utc
        """, (
            STOCKS_TO_EXPLORE[0],
            STOCKS_TO_EXPLORE[1],
            START_DATE,
            END_DATE
        ))

        total = 0
        stock_counts = {"HSY": 0, "MCD": 0}

        # Create a separate CSV file.
        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow([
                "symbol",
                "ts_utc",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ])

            # Retrieve records in batches.
            while True:

                rows = cursor.fetchmany(1000)

                if not rows:
                    break

                writer.writerows(rows)

                total += len(rows)

                for row in rows:
                    stock_counts[row[0]] += 1

    print("\n=== CSV FILE CREATED ===")
    print("File:", OUTPUT_FILE)
    print("HSY records:", stock_counts["HSY"])
    print("MCD records:", stock_counts["MCD"])
    print("Total records:", total)


if __name__ == "__main__":
    main()
