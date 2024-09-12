import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data_processing import load_data

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


def train(max_features=10_000, C=1.0):
    """
    Train a Logistic Regression classifier on TF-IDF features and save the model.

    Uses the SAGA solver with L1 regularization. SAGA is chosen over lbfgs because
    it handles large sparse feature matrices efficiently via stochastic gradient updates,
    and is the only solver that supports L1 which performs implicit feature selection.

    Args:
        max_features (int): Number of TF-IDF features passed to load_data(). Default 10,000.
        C (float): Inverse of regularization strength — smaller values = stronger regularization.

    Returns:
        model: the fitted LogisticRegression instance

    Example:
        $ python train.py
        Training with max_features=10000, C=1.0
        Test F1: 0.8741
        Model saved to model.joblib
        Vectorizer saved to vectorizer.joblib
    """
    X_train, X_test, y_train, y_test, vectorizer = load_data(max_features=max_features)

    model = LogisticRegression(
        solver="saga",      # stochastic solver — scales well to large sparse TF-IDF matrices
        penalty="l1",       # L1 drives irrelevant feature weights to zero (implicit feature selection)
        C=C,                # inverse regularization strength — lower C = stronger penalty on large weights
        max_iter=1000,      # SAGA needs more iterations than lbfgs to converge on large datasets
        n_jobs=-1,          # use all available CPU cores
    )

    print(f"Training with max_features={max_features}, C={C}")
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
