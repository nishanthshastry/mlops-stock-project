import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlops_stock_project.config import (
    PROCESSED_DATA_FILE
)

from mlops_stock_project.logging_config import (
    get_logger
)

from mlops_stock_project.visualization.visualize import (
    save_accuracy_plot
)

logger = get_logger(__name__)


def simulate_baseline(
    data_path=PROCESSED_DATA_FILE
):
    logger.info("Running baseline simulation...")

    df = pd.read_csv(data_path)

    features = [
        "Return",
        "MA_5",
        "MA_10",
        "Volatility"
    ]

    split_index = int(len(df) * 0.6)

    train_data = df[:split_index]

    test_data = df[split_index:]

    model = LogisticRegression()

    model.fit(
        train_data[features],
        train_data["Target"]
    )

    accuracies = []

    window_size = 20

    for i in range(window_size, len(test_data)):
        window = test_data.iloc[:i]

        y_true = window["Target"]

        y_pred = model.predict(window[features])

        acc = accuracy_score(y_true, y_pred)

        accuracies.append(acc)

    save_accuracy_plot(
        accuracies,
        title="Baseline Model Accuracy Over Time",
        filename="baseline_accuracy.png"
    )

    logger.info("Baseline simulation completed")

    return accuracies


if __name__ == "__main__":
    simulate_baseline()