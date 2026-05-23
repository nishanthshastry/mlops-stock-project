import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def simulate_drift_retraining(data_path="data/processed/AAPL_features.csv", threshold=0.52):
    print("Simulating drift-based retraining...")

    df = pd.read_csv(data_path)

    features = ["Return", "MA_5", "MA_10", "Volatility"]

    split_index = int(len(df) * 0.6)

    train_data = df[:split_index]
    test_data = df[split_index:]

    model = LogisticRegression()
    model.fit(train_data[features], train_data["Target"])

    accuracies = []
    window_size = 20

    for i in range(window_size, len(test_data)):
        window = test_data.iloc[:i]

        y_true = window["Target"]
        y_pred = model.predict(window[features])

        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)

        # Drift condition → retrain
        if acc < threshold:
            retrain_data = df[:split_index + i]
            model.fit(retrain_data[features], retrain_data["Target"])

    # Plot
    plt.plot(accuracies)
    plt.title("Drift-Based Retraining Accuracy Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Accuracy")
    plt.show()

    return accuracies


if __name__ == "__main__":
    simulate_drift_retraining()