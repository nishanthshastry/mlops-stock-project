from sklearn.metrics import accuracy_score

def evaluate_classification_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": accuracy
    }