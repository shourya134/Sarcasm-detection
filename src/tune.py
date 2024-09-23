import os
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .data_processing import clean_text, DATA_DIR

TUNING_RESULTS_DIR = "tuning_results"
MARGINAL_PLOT_PATH = os.path.join(TUNING_RESULTS_DIR, "f1_vs_hyperparameters.png")
HEATMAP_PLOT_PATH = os.path.join(TUNING_RESULTS_DIR, "f1_combination_heatmap.png")

PARAM_GRID = {
    "tfidf__max_features": [5_000, 10_000, 15_000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [0.1, 1, 10],
    "clf__penalty": ["l1", "l2"],
    "clf__class_weight": [None, "balanced"],
}
CV_FOLDS = 3


def tune():
    """
    Grid search TF-IDF + LogisticRegression hyperparameters jointly via cross-validation.

    Searches max_features, ngram_range, C, penalty, and class_weight together in a
    single Pipeline so vectorizer and model hyperparameters are evaluated as combinations,
    not independently. Saves two plots:
      - f1_vs_hyperparameters.png: marginal effect of each hyperparameter (mean CV F1
        across all combinations that use that value), showing each param's individual trend
      - f1_combination_heatmap.png: F1 for combinations of the two hyperparameters with the
        widest marginal F1 range, holding the rest at the overall best values — reveals
        whether the best combination differs from stacking each param's individual best

    Example:
        $ python -m src.tune
        Fitting 3 folds for each of 108 candidates, totalling 324 fits
        Best params: {'clf__C': 10, 'clf__class_weight': None, 'clf__penalty': 'l2', ...}
        Best CV F1: 0.8781
        Marginal effect plot saved to tuning_results/f1_vs_hyperparameters.png
        Combination heatmap saved to tuning_results/f1_combination_heatmap.png
    """
    os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    train_df["headline"] = train_df["headline"].map(clean_text)
    X_train, y_train = train_df["headline"], train_df["is_sarcastic"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(solver="saga", max_iter=1000, n_jobs=-1)),
    ])

    search = GridSearchCV(
        pipeline, PARAM_GRID, scoring="f1", cv=CV_FOLDS, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    results = pd.DataFrame(search.cv_results_)
    _plot_marginal_effects(results)
    _plot_combination_heatmap(results, search.best_params_)

    return search


def _plot_marginal_effects(results):
    """For each hyperparameter, plot mean CV F1 per value, averaged over all other params."""
    param_names = list(PARAM_GRID.keys())
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))

    for ax, param in zip(axes, param_names):
        # dropna=False: class_weight=None is a real grid value, not missing data —
        # pandas groupby drops NaN-like keys by default, which would silently exclude it
        grouped = results.groupby(f"param_{param}", dropna=False)["mean_test_score"].mean()
        labels = [str(v) for v in grouped.index]
        ax.plot(labels, grouped.values, marker="o")
        ax.set_xlabel(param.split("__")[1])
        ax.set_ylabel("Mean CV F1")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)

    fig.suptitle("Marginal effect of each hyperparameter on CV F1")
    fig.tight_layout()
    fig.savefig(MARGINAL_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Marginal effect plot saved to {MARGINAL_PLOT_PATH}")


def _plot_combination_heatmap(results, best_params):
    """
    Heatmap of F1 for the two hyperparameters with the widest marginal F1 range,
    holding all other params fixed at their best values from the search.

    The two widest-range params are chosen because they individually influence F1
    the most, making their interaction the most informative to visualize in 2D.
    """
    param_names = list(PARAM_GRID.keys())
    ranges = {
        param: results.groupby(f"param_{param}", dropna=False)["mean_test_score"].mean().max()
        - results.groupby(f"param_{param}", dropna=False)["mean_test_score"].mean().min()
        for param in param_names
    }
    param_x, param_y = sorted(ranges, key=ranges.get, reverse=True)[:2]

    fixed_params = {p: best_params[p] for p in param_names if p not in (param_x, param_y)}
    mask = pd.Series(True, index=results.index)
    for param, value in fixed_params.items():
        # fillna: value may be None (a real grid value, e.g. class_weight=None), and
        # NaN == None is always False in pandas — comparing filled sentinels matches correctly
        mask &= results[f"param_{param}"].fillna("__none__") == (value if value is not None else "__none__")
    subset = results[mask]

    x_vals = PARAM_GRID[param_x]
    y_vals = PARAM_GRID[param_y]
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    # fillna as above: xv/yv may be None (e.g. class_weight=None) if param_x/param_y
    # happens to be class_weight, so compare against filled columns to match correctly
    x_col = subset[f"param_{param_x}"].fillna("__none__")
    y_col = subset[f"param_{param_y}"].fillna("__none__")
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            xv_key = xv if xv is not None else "__none__"
            yv_key = yv if yv is not None else "__none__"
            row = subset[(x_col == xv_key) & (y_col == yv_key)]
            if not row.empty:
                grid[i, j] = row["mean_test_score"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(param_x.split("__")[1])
    ax.set_ylabel(param_y.split("__")[1])
    ax.set_title(
        f"CV F1 by {param_x.split('__')[1]} x {param_y.split('__')[1]}\n"
        f"(other params fixed at best: {fixed_params})"
    )
    for i, j in itertools.product(range(len(y_vals)), range(len(x_vals))):
        if not np.isnan(grid[i, j]):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Mean CV F1")
    fig.tight_layout()
    fig.savefig(HEATMAP_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Combination heatmap saved to {HEATMAP_PLOT_PATH}")


if __name__ == "__main__":
    tune()
