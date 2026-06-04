import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlops_stock_project.config import FIGURES_DIR
from mlops_stock_project.logging_config import get_logger

logger = get_logger(__name__)


def save_accuracy_plot(accuracies, title, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.plot(accuracies)

    plt.title(title)

    plt.xlabel("Time Steps")

    plt.ylabel("Accuracy")

    output_path = FIGURES_DIR / filename

    plt.savefig(output_path)

    logger.info(f"Figure saved to {output_path}")

    plt.close()
