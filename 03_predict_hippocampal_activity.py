#!/usr/bin/env python3
"""Predict subject-level hippocampal activity from cortical encoding scores.

For each peri-event time bin, fit a linear SVR with leave-one-subject-out
cross-validation. Standardization is learned separately within each training
fold. An empirical one-sided p-value is obtained by permuting targets among
the same eligible subjects and refitting the complete cross-validation
procedure. Benjamini-Hochberg FDR correction is applied across time bins.

The participant order in the encoding-score and hippocampal arrays MUST match.
The cortical ROI sets must be defined independently of the tested outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.multitest import multipletests


FEATURE_NAMES = ("IFG", "ITG", "MFG")


def load_roi_indices(path: Path, n_rois: int, one_based: bool) -> np.ndarray:
    """Validate an ROI index file and return unique zero-based indices."""
    raw = np.asarray(np.load(path, allow_pickle=False)).ravel()
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError(f"Empty or non-finite ROI indices: {path}")
    if not np.all(raw == np.floor(raw)):
        raise ValueError(f"Non-integer ROI indices: {path}")
    indices = raw.astype(int) - int(one_based)
    if (indices < 0).any() or (indices >= n_rois).any():
        raise ValueError(f"ROI indices out of bounds in {path}")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Duplicate ROI indices in {path}")
    return indices


def build_cortical_features(
    scores: np.ndarray, roi_files: list[Path], one_based: bool
) -> np.ndarray:
    """Average encoding scores across ROIs for IFG, ITG, and MFG."""
    if scores.ndim != 2:
        raise ValueError("Encoding scores must have shape (subject, cortical ROI).")
    columns = []
    for path in roi_files:
        indices = load_roi_indices(path, scores.shape[1], one_based)
        columns.append(scores[:, indices].mean(axis=1))
    return np.column_stack(columns)


def loso_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return out-of-fold SVR predictions for an already-filtered dataset."""
    predicted = np.full(y.shape, np.nan, dtype=float)
    for train_index, test_index in LeaveOneOut().split(X):
        model = make_pipeline(
            StandardScaler(),
            SVR(kernel="linear", C=1.0, epsilon=0.1),
        )
        model.fit(X[train_index], y[train_index])
        predicted[test_index] = model.predict(X[test_index])
    return predicted


def prediction_correlation(y: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Pearson r, returning NaN for insufficient/constant data."""
    if len(y) < 3 or np.std(y) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(pearsonr(y, predicted).statistic)


def permutation_test(
    X: np.ndarray, y: np.ndarray, n_permutations: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    """Run LOSO prediction and a fixed-sample, one-sided permutation test.

    Returns original subject indices, valid targets, predictions, observed
    Pearson r, descriptive Pearson p, empirical permutation p, and null r.
    The descriptive Pearson p does not account for cross-validation.
    """
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    subject_indices = np.flatnonzero(valid)
    X_valid, y_valid = X[valid], y[valid]
    null = np.full(n_permutations, np.nan)

    if len(y_valid) < 3 or np.std(y_valid) == 0:
        return (subject_indices, y_valid, np.full(y_valid.shape, np.nan),
                np.nan, np.nan, np.nan, null)

    prediction = loso_predictions(X_valid, y_valid)
    observed_r = prediction_correlation(y_valid, prediction)
    descriptive_p = (
        float(pearsonr(y_valid, prediction).pvalue)
        if np.isfinite(observed_r) else np.nan
    )

    if not np.isfinite(observed_r):
        return (subject_indices, y_valid, prediction,
                np.nan, descriptive_p, np.nan, null)

    rng = np.random.default_rng(seed)
    for permutation_index in range(n_permutations):
        shuffled = rng.permutation(y_valid)
        null_prediction = loso_predictions(X_valid, shuffled)
        null[permutation_index] = prediction_correlation(shuffled, null_prediction)

    finite_null = null[np.isfinite(null)]
    empirical_p = (
        (np.count_nonzero(finite_null >= observed_r) + 1)
        / (finite_null.size + 1)
        if finite_null.size else np.nan
    )
    return (
        subject_indices, y_valid, prediction, observed_r,
        descriptive_p, empirical_p, null,
    )


def plot_performance(results: pd.DataFrame, output_path: Path) -> None:
    """Plot cross-validated prediction correlation and FDR annotations."""
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, results["r_prediction"], color="black", marker="o", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    labels = results["timepoint"].tolist()
    if "During" in labels:
        middle = labels.index("During")
        ax.axvspan(middle - 0.4, middle + 0.4, color="gray", alpha=0.15)

    for i, row in results.iterrows():
        p = row["p_perm_fdr"]
        r = row["r_prediction"]
        if np.isfinite(p) and np.isfinite(r):
            marker = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            ax.annotate(marker, (i, r), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xticks(x, labels, rotation=15)
    ax.set_xlabel("Relative time")
    ax.set_ylabel("LOSO prediction performance (Pearson r)")
    ax.set_title("Prediction of left hippocampal activity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding-scores", type=Path, required=True,
                        help="Subject x cortical-ROI encoding scores (.npy)")
    parser.add_argument("--ifg-indices", type=Path, required=True)
    parser.add_argument("--itg-indices", type=Path, required=True)
    parser.add_argument("--mfg-indices", type=Path, required=True)
    parser.add_argument("--hippocampus", type=Path, required=True,
                        help="Subject x relative-time hippocampal activity (.npy)")
    parser.add_argument("--timepoints", type=Path, required=True,
                        help="Ordered relative-time labels (.npy)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--one-based-indices", action="store_true",
                        help="Convert ROI IDs starting at 1 to Python indices.")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("--n-permutations must be at least 1.")

    scores = np.load(args.encoding_scores, allow_pickle=False)
    X = build_cortical_features(
        scores,
        [args.ifg_indices, args.itg_indices, args.mfg_indices],
        args.one_based_indices,
    )
    Y = np.load(args.hippocampus, allow_pickle=False)
    labels = np.load(args.timepoints, allow_pickle=False).tolist()
    if Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("The encoding and hippocampal arrays must have "
                         "matching subject dimensions.")
    if not isinstance(labels, list) or len(labels) != Y.shape[1]:
        raise ValueError("Timepoint labels must match the hippocampal columns.")
    if len(set(labels)) != len(labels):
        raise ValueError("Timepoint labels must be unique.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "SVR_cortical_features_IFG_ITG_MFG.npy", X)
    np.save(args.output_dir / "SVR_cortical_feature_names.npy",
            np.asarray(FEATURE_NAMES))
    np.save(args.output_dir / "SVR_hippo_L_subject_time.npy", Y)
    np.save(args.output_dir / "timepoints_order.npy", np.asarray(labels))

    rows = []
    null_distributions = []
    for column, label in enumerate(labels):
        (indices, observed, predicted, r, descriptive_p, permutation_p,
         null) = permutation_test(X, Y[:, column],
                                  args.n_permutations, args.seed + column)
        rows.append({
            "timepoint": label,
            "timepoint_index": column,
            "n_subjects": len(indices),
            "r_prediction": r,
            "p_pearson_descriptive": descriptive_p,
            "p_perm": permutation_p,
            "null_mean": np.nanmean(null) if np.isfinite(null).any() else np.nan,
            "null_std": np.nanstd(null) if np.isfinite(null).any() else np.nan,
        })
        pd.DataFrame({
            "subject_original_index": indices,
            "observed_hippo_L": observed,
            "predicted_hippo_L": predicted,
        }).to_csv(args.output_dir / f"prediction_observed_{label}.csv", index=False)
        null_distributions.append(null)
        print(f"{label}: n={len(indices)}, r={r:.4f}, p_perm={permutation_p:.4g}")

    results = pd.DataFrame(rows)
    results["p_perm_fdr"] = np.nan
    results["sig_fdr_0.05"] = False
    valid_p = np.isfinite(results["p_perm"].to_numpy())
    if valid_p.any():
        reject, adjusted, _, _ = multipletests(
            results.loc[valid_p, "p_perm"], alpha=0.05, method="fdr_bh"
        )
        results.loc[valid_p, "p_perm_fdr"] = adjusted
        results.loc[valid_p, "sig_fdr_0.05"] = reject

    results.to_csv(
        args.output_dir / "SVR_cortical_encoding_predict_hippo_L_time_resolved_results.csv",
        index=False,
    )
    np.save(
        args.output_dir / "SVR_null_distribution_timepoint_by_perm.npy",
        np.stack(null_distributions),
    )
    plot_performance(
        results,
        args.output_dir / "SVR_cortical_encoding_predict_hippo_L_time_resolved_r.png",
    )
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
