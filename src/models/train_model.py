import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(data_path="data/processed/AAPL_features.csv"):
    print("Training model...")

    df = pd.read_csv(data_path)

    # Features and target
    features = ["Return", "MA_5", "MA_10", "Volatility"]
    X = df[features]
    y = df["Target"]

    # Train-test split (time-based split)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")

    return model, accuracy


if __name__ == "__main__":
    model, acc = train_model()