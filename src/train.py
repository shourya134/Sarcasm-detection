import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data_processing import load_data

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.joblib")


def train(max_features=5_000, C=10.0, penalty="l2", class_weight="balanced", ngram_range=(1, 3)):
    """
    Train a Logistic Regression classifier on TF-IDF features and save the model.

    Uses the SAGA solver, which scales well to large sparse TF-IDF matrices via
    stochastic gradient updates and supports both L1 and L2 penalties.

    Args:
        max_features (int): Number of TF-IDF features passed to load_data(). Default 5,000.
        C (float): Inverse of regularization strength — smaller values = stronger regularization.
        penalty (str): Regularization type ("l1" or "l2").
        class_weight (str or None): Passed to LogisticRegression, e.g. "balanced".
        ngram_range (tuple): n-gram range passed to load_data().

    Returns:
        model: the fitted LogisticRegression instance

    Example:
        $ python train.py
        Training with max_features=5000, C=10.0, penalty=l2, class_weight=balanced, ngram_range=(1, 3)
        Test F1: 0.9444
        Model saved to artifacts/model.joblib
        Vectorizer saved to artifacts/vectorizer.joblib
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, vectorizer = load_data(
        max_features=max_features, ngram_range=ngram_range
    )

    model = LogisticRegression(
        solver="saga",              # stochastic solver — scales well to large sparse TF-IDF matrices
        penalty=penalty,
        C=C,                        # inverse regularization strength — lower C = stronger penalty on large weights
        class_weight=class_weight,
        max_iter=1000,              # SAGA needs more iterations than lbfgs to converge on large datasets
        n_jobs=-1,                  # use all available CPU cores
    )

    print(f"Training with max_features={max_features}, C={C}, penalty={penalty}, "
          f"class_weight={class_weight}, ngram_range={ngram_range}")
    model.fit(X_train, y_train)

    f1 = f1_score(y_test, model.predict(X_test))
    print(f"Test F1: {f1:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Saved alongside the model — needed to transform any new raw text the same way at inference time
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return model


if __name__ == "__main__":
    train()
