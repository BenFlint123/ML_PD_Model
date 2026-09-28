"""Regenerate the champion vs challenger ROC figure and metrics used in README.md.

Reads the train / test split written by notebook 02, refits both models with
the same settings as notebooks 03 (champion) and 04 (challenger), and writes
``docs/images/roc_comparison.png``.

Usage:
    uv run python scripts/make_readme_figure.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CAT_VARS, DATA, NUM_VARS, RANDOM_SEED, ROOT
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib.model_development import LightGbmModelBuilder


OUTPUT = ROOT / "docs" / "images" / "roc_comparison.png"

# Categorical slots 1 and 2 of the reference data-viz palette.
CHAMPION_COLOUR = "#2a78d6"
CHALLENGER_COLOUR = "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fit_champion(X_train, y_train, X_test) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression, as in notebook 03."""
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUM_VARS),
                        (
                            "cat",
                            OneHotEncoder(drop="first", handle_unknown="ignore"),
                            CAT_VARS,
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=np.inf)),
        ]
    ).fit(X_train, y_train)
    return model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1]


def fit_challenger(X_train, y_train, X_test, y_test) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM with default parameters, as in notebook 04."""
    results = LightGbmModelBuilder.from_split_data(
        X_train, y_train, X_test, y_test, {"verbose": -1}, random_state=RANDOM_SEED
    ).run()
    return results["pred_train"], results["pred_test"]


def gini(y, score) -> float:
    return 2 * roc_auc_score(y, score) - 1


def main() -> None:
    split_dir = DATA / "processed" / "basic_data_prep"
    d = {f.stem: pd.read_csv(f) for f in split_dir.glob("*.csv")}
    y_train = d["y_train"].to_numpy().ravel()
    y_test = d["y_test"].to_numpy().ravel()

    models = {
        "Champion: logistic regression": (
            fit_champion(d["X_train"], y_train, d["X_test"]),
            CHAMPION_COLOUR,
        ),
        "Challenger: LightGBM": (
            fit_challenger(d["X_train"], y_train, d["X_test"], y_test),
            CHALLENGER_COLOUR,
        ),
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--", label="Random (Gini 0)")

    print(f"{'Model':32} {'Train Gini':>10} {'Test Gini':>10} {'Test AUC':>9}")
    for name, ((pred_train, pred_test), colour) in models.items():
        g_train, g_test = gini(y_train, pred_train), gini(y_test, pred_test)
        print(
            f"{name:32} {g_train:10.3f} {g_test:10.3f} "
            f"{roc_auc_score(y_test, pred_test):9.3f}"
        )
        fpr, tpr, _ = roc_curve(y_test, pred_test)
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (Gini {g_test:.2f})")

    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
    ax.set_xlabel("False positive rate", color=INK_MUTED)
    ax.set_ylabel("True positive rate", color=INK_MUTED)
    ax.set_title(
        "ROC on the hold-out test set (n = 6,000)", color=INK, loc="left", pad=10
    )
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=INK_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK, fontsize=9)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
