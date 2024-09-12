import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from data_processing import clean_text, DATA_DIR

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"


def evaluate():
    """
    Load the trained model, run it on the held-out test set, and report metrics.

    Prints a precision/recall/F1 classification report and saves a confusion
    matrix plot to disk for inclusion in project documentation.

    Example:
        $ python evaluate.py
                      precision    recall  f1-score   support
                   0       0.87      0.89      0.88     14985
                   1       0.87      0.85      0.86     11724
        Confusion matrix saved to confusion_matrix.png
    """
    model = joblib.load(MODEL_PATH)
    # Load the vectorizer fit during training — guarantees the exact same vocabulary,
    # rather than risking a mismatch from re-fitting a new one here
    vectorizer = joblib.load(VECTORIZER_PATH)

    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    test_df["headline"] = test_df["headline"].map(clean_text)

    X_test = vectorizer.transform(test_df["headline"])
    y_test = test_df["is_sarcastic"]

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=["not sarcastic", "sarcastic"]))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["not sarcastic", "sarcastic"], cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate()
