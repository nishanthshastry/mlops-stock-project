import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def simulate_baseline(data_path="data/processed/AAPL_features.csv"):
    print("Simulating baseline model...")

    df = pd.read_csv(data_path)

    features = ["Return", "MA_5", "MA_10", "Volatility"]

    split_index = int(len(df) * 0.6)

    train_data = df[:split_index]
    test_data = df[split_index:]

    # Train once (baseline)
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

    # Plot
    plt.plot(accuracies)
    plt.title("Baseline Model Accuracy Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Accuracy")
    plt.show()

    return accuracies


if __name__ == "__main__":
    simulate_baseline()