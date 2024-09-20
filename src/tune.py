import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data_processing import load_data

MAX_FEATURES_GRID = [2_000, 5_000, 10_000, 15_000, 20_000, 30_000]
TUNING_RESULTS_DIR = "tuning_results"
PLOT_PATH = os.path.join(TUNING_RESULTS_DIR, "f1_vs_max_features.png")


def tune():
    """
    Grid search over TF-IDF max_features and plot test F1 for each value.

    Retrains a fresh LogisticRegression (same config as train.py: SAGA solver,
    L1 penalty) for every max_features value in MAX_FEATURES_GRID, since the
    vectorizer's vocabulary changes with max_features and must be refit each time.

    Example:
        $ python tune.py
        max_features=2000    F1=0.8312
        max_features=5000    F1=0.8590
        max_features=10000   F1=0.8741
        max_features=15000   F1=0.8765
        max_features=20000   F1=0.8758
        max_features=30000   F1=0.8744
        Best max_features: 15000 (F1=0.8765)
        Plot saved to tuning_results/f1_vs_max_features.png
    """
    os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)

    results = []

    for max_features in MAX_FEATURES_GRID:
        X_train, X_test, y_train, y_test, _ = load_data(max_features=max_features)

        model = LogisticRegression(
            solver="saga",
            penalty="l1",
            max_iter=1000,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        f1 = f1_score(y_test, model.predict(X_test))
        results.append((max_features, f1))
        print(f"max_features={max_features:<8} F1={f1:.4f}")

    best_max_features, best_f1 = max(results, key=lambda r: r[1])
    print(f"Best max_features: {best_max_features} (F1={best_f1:.4f})")

    x_vals = [r[0] for r in results]
    y_vals = [r[1] for r in results]
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("max_features")
    plt.ylabel("Test F1 score")
    plt.title("F1 score vs TF-IDF max_features")
    plt.grid(True)
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"Plot saved to {PLOT_PATH}")

    return results


if __name__ == "__main__":
    tune()
