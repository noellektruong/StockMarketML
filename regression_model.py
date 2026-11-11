
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

FEATURES = ["close", "volume", "price_range"]
TARGET = "future_close"

TRAIN_FILE = "stock_train.csv"
TEST_FILE = "stock_test.csv"


def main():

    # Load training and testing data
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    all_predictions = []
    models = {}

    # Train a separate model for each stock
    for stock in ["HSY", "MCD"]:

        stock_train = train_data[train_data["symbol"] == stock]
        stock_test = test_data[test_data["symbol"] == stock].copy()

        X_train = stock_train[FEATURES]
        y_train = stock_train[TARGET]

        X_test = stock_test[FEATURES]

        # Train the regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        models[stock] = model

        # Generate predictions
        stock_test["prediction"] = model.predict(X_test)

        all_predictions.append(stock_test)

    # Combine predictions from both stocks
    results = pd.concat(all_predictions, ignore_index=True)

    # Calculate overall model performance
    mae = mean_absolute_error(
        results["future_close"],
        results["prediction"]
    )

    r2 = r2_score(
        results["future_close"],
        results["prediction"]
    )

    # Baseline: predict future price equals current price
    baseline_mae = mean_absolute_error(
        results["future_close"],
        results["close"]
    )

    baseline_r2 = r2_score(
        results["future_close"],
        results["close"]
    )

    print("\nModel Results")
    print("----------------")
    print("Mean Absolute Error:", round(mae, 4))
    print("R-squared:", round(r2, 4))

    print("\nBaseline Results")
    print("----------------")
    print("Baseline MAE:", round(baseline_mae, 4))
    print("Baseline R-squared:", round(baseline_r2, 4))

    # Results by stock
    print("\nResults By Stock")
    print("----------------")

    for stock in ["HSY", "MCD"]:

        stock_results = results[
            results["symbol"] == stock
        ]

        print(f"\n{stock}")

        print("MAE:", round(mean_absolute_error(
            stock_results["future_close"],
            stock_results["prediction"]
        ), 4))

        print("R-squared:", round(r2_score(
            stock_results["future_close"],
            stock_results["prediction"]
        ), 4))

    # Display 20 sample predictions
    sample = results.sample(
        n=min(20, len(results)),
        random_state=42
    ).reset_index(drop=True)

    predictions = sample[
        ["symbol", "close", "volume", "price_range",
         "future_close", "prediction"]
    ].copy()

    print("\nSample Predictions")
    print(predictions.to_string())

    # Generate buy signals
    buy_signals = sample[
        ["symbol", "close", "prediction"]
    ].copy()

    buy_signals["buy_signal"] = (
        buy_signals["prediction"] > buy_signals["close"]
    )

    print("\nBuy Signals")
    print(buy_signals.to_string())

    # Save all test predictions
    results.to_csv("stock_predictions.csv", index=False)

    print("\nSaved predictions to stock_predictions.csv")

    # ==========================================
    # VISUALIZATION 1: ACTUAL VS PREDICTED PRICES
    # ==========================================

    for stock in ["HSY", "MCD"]:

        stock_results = results[
            results["symbol"] == stock
        ].sort_values("ts_utc").head(200)

        plt.figure(figsize=(12, 6))

        plt.plot(
            range(len(stock_results)),
            stock_results["future_close"],
            label="Actual Future Price"
        )

        plt.plot(
            range(len(stock_results)),
            stock_results["prediction"],
            label="Predicted Future Price"
        )

        plt.title(f"{stock}: Actual vs Predicted Prices")
        plt.xlabel("Test Observations")
        plt.ylabel("Stock Price ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"{stock}_actual_vs_predicted.png")
        plt.show()

    # ==========================================
    # VISUALIZATION 2: PREDICTION ERRORS
    # ==========================================

    results["error"] = (
        results["future_close"] - results["prediction"]
    )

    plt.figure(figsize=(10, 6))

    plt.hist(results["error"], bins=40)

    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Prediction Error ($)")
    plt.ylabel("Number of Predictions")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("prediction_errors.png")
    plt.show()

    # ==========================================
    # VISUALIZATION 3: MODEL VS BASELINE
    # ==========================================

    plt.figure(figsize=(8, 6))

    model_names = ["Regression", "Baseline"]
    errors = [mae, baseline_mae]

    plt.bar(model_names, errors)

    plt.title("Regression vs Baseline MAE")
    plt.ylabel("Mean Absolute Error ($)")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig("model_comparison.png")
    plt.show()

    # ==========================================
    # VISUALIZATION 4: FEATURE COEFFICIENTS
    # ==========================================

    for stock in ["HSY", "MCD"]:

        model = models[stock]

        plt.figure(figsize=(8, 6))

        plt.bar(FEATURES, model.coef_)

        plt.title(f"{stock}: Regression Coefficients")
        plt.xlabel("Input Variables")
        plt.ylabel("Coefficient Value")
        plt.axhline(y=0, linewidth=0.8)
        plt.grid(axis="y")
        plt.tight_layout()

        plt.savefig(f"{stock}_feature_coefficients.png")
        plt.show()

    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    main()