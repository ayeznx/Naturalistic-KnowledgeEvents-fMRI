#!/usr/bin/env python3
"""Fit a five-feature, subject-wise cortical encoding model.

This script reorganizes the original encoding workflow while preserving its
random TR-level split, HRF-convolution approach, calls to the original
encoding/statistics helpers, feature-group weight contributions, and output
filenames. See README_encoding_model.md for methodological qualifications.

External files REQUIRED:
    encoding_helpers.py
    fdr_correction_helpers.py
Their original implementations are needed to reproduce the published model
fitting, null distributions, p-values, and FDR thresholding.
"""

from __future__ import annotations

import os

# Set thread limits before importing NumPy/joblib in the main process.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import importlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


FEATURE_NAMES = ("vis", "aud", "cha", "sem", "know")
SCORE_FILENAMES = {
    "full": "encoding_results_full_scores_unthreshold.npy",
    "vis": "encoding_results_f1_vis_scores_unthreshold.npy",
    "aud": "encoding_results_f2_aud_scores_unthreshold.npy",
    "cha": "encoding_results_f3_cha_scores_unthreshold.npy",
    "sem": "encoding_results_f4_sem_scores_unthreshold.npy",
    "know": "encoding_results_f5_know_scores_unthreshold.npy",
}
NULL_PREFIXES = {
    "full": "all",
    "vis": "f1_vis",
    "aud": "f2_aud",
    "cha": "f3_cha",
    "sem": "f4_sem",
    "know": "f5_know",
}
RESULT_PREFIXES = {
    "full": "full",
    "vis": "f1_vis",
    "aud": "f2_aud",
    "cha": "f3_cha",
    "sem": "f4_sem",
    "know": "f5_know",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bold", required=True, type=Path,
                        help="BOLD array: (subjects, cortical ROIs, TRs)")
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--domain", default="all",
                        help="Feature filename suffix; default: all")
    parser.add_argument("--knowledge-prefix", default="know_8",
                        help="Knowledge filename prefix; default: know_8")
    parser.add_argument("--encoding-helpers-dir", type=Path,
                        help="Folder containing the ORIGINAL encoding_helpers.py")
    parser.add_argument("--fdr-helpers-dir", type=Path,
                        help="Folder containing the ORIGINAL fdr_correction_helpers.py")
    parser.add_argument("--tr", type=float, default=2.0, help="TR, in seconds")
    parser.add_argument("--hrf-oversampling", type=int, default=16)
    parser.add_argument("--hrf-duration", type=float, default=32.0,
                        help="Duration of the HRF kernel, in seconds")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--null-seed", type=int, default=None,
                        help="Optional seed for helpers using NumPy's legacy "
                             "global RNG; other RNGs may not be controlled")
    args = parser.parse_args()
    if args.tr <= 0 or args.hrf_oversampling < 1 or args.hrf_duration <= 0:
        parser.error("TR, oversampling and HRF duration must be positive.")
    if not (0 < args.test_size < 1):
        parser.error("--test-size must be strictly between 0 and 1.")
    if args.n_permutations < 1 or args.n_jobs == 0:
        parser.error("--n-permutations must be >= 1 and --n-jobs cannot be 0.")
    if not (0 < args.fdr_alpha < 1):
        parser.error("--fdr-alpha must be strictly between 0 and 1.")
    return args


def import_project_helper(module_name: str, folder: Path | None) -> Any:
    """Import the user's original helper without silently replacing its method."""
    if folder is not None:
        path = folder.resolve() / f"{module_name}.py"
        if not path.is_file():
            raise FileNotFoundError(f"Required original helper not found: {path}")
        sys.path.insert(0, str(folder.resolve()))
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing {module_name}.py. Supply the original implementation "
            f"using its corresponding --*-helpers-dir argument. "
            "This script does not substitute an unverified algorithm."
        ) from exc


def load_2d_feature(path: Path, expected_trs: int) -> np.ndarray:
    """Load an (TR, feature) array, accepting a 1D single-feature vector."""
    data = np.load(path, allow_pickle=False)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] != expected_trs or data.shape[1] == 0:
        raise ValueError(
            f"{path}: expected (TR={expected_trs}, n_features>=1), "
            f"got {data.shape}."
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{path} contains NaN or infinite feature values.")
    return np.asarray(data, dtype=float)


def spm_style_hrf(
    tr: float,
    oversampling: int = 16,
    duration: float = 32.0,
    peak1: float = 6.0,
    under1: float = 16.0,
    undershoot_ratio: float = 1 / 6,
    onset: float = 0.0,
) -> np.ndarray:
    """Sample the original double-gamma-style HRF expression at each TR.

    For reproduction, this intentionally retains the original formula:
    gamma densities with unit scale; fine-grid normalization followed by
    taking every `oversampling`-th sample, without TR-grid renormalization.
    It is 'SPM-style', not a claim of numerical identity to SPM's spm_hrf().
    """
    from math import gamma

    fine_time = np.arange(0, duration, tr / oversampling)
    shifted = fine_time - onset
    t_nonnegative = np.maximum(shifted, 0.0)

    def gamma_pdf(time: np.ndarray, shape: float) -> np.ndarray:
        return time ** (shape - 1) * np.exp(-time) / gamma(shape)

    fine_hrf = (
        gamma_pdf(t_nonnegative, peak1)
        - undershoot_ratio * gamma_pdf(t_nonnegative, under1)
    )
    fine_hrf[shifted < 0] = 0.0
    normalization = np.sum(fine_hrf) + 1e-12
    if abs(normalization) < 1e-10:
        raise ValueError("HRF normalization is numerically unstable.")
    return fine_hrf[::oversampling] / normalization


def convolve_and_zscore(
    features: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """Convolve and standardize each feature over *all* TRs, as originally.

    The original workflow standardizes before splitting train/test; this
    is retained for comparability, and its potential leakage is documented.
    """
    n_trs, n_features = features.shape
    result = np.empty((n_trs, n_features), dtype=float)
    for column in range(n_features):
        result[:, column] = np.convolve(
            features[:, column], kernel, mode="full"
        )[:n_trs]
    means = result.mean(axis=0)
    stds = result.std(axis=0, ddof=0)
    result = (result - means) / np.where(stds == 0, 1.0, stds)
    return result


def load_design(
    feature_dir: Path,
    domain: str,
    knowledge_prefix: str,
    n_trs: int,
    kernel: np.ndarray,
) -> tuple[OrderedDict[str, np.ndarray], dict[str, slice], np.ndarray]:
    """Construct design matrix in the original feature-group order."""
    filenames = {
        "vis": f"vis_{domain}.npy",
        "aud": f"aud_{domain}.npy",
        "cha": f"cha_{domain}_new.npy",
        "sem": f"sem_{domain}_word2vec.npy",
        "know": f"{knowledge_prefix}_{domain}.npy",
    }
    groups: OrderedDict[str, np.ndarray] = OrderedDict()
    slices: dict[str, slice] = {}
    start = 0
    for name in FEATURE_NAMES:
        array = load_2d_feature(feature_dir / filenames[name], n_trs)
        convolved = convolve_and_zscore(array, kernel)
        groups[name] = convolved
        stop = start + convolved.shape[1]
        slices[name] = slice(start, stop)
        start = stop
    return groups, slices, np.concatenate(list(groups.values()), axis=1)


def process_one_subject(
    subject: np.ndarray,
    all_test: np.ndarray,
    all_train: np.ndarray,
    all_features: list[np.ndarray],
    train_features: list[np.ndarray],
    test_features: list[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_slices: dict[str, slice],
    encoding_helpers: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit original helper model and evaluate full/group weight contributions.

    Group scores reflect a slice of the FULL fitted model weights; they
    are not scores from independently retrained single-feature models.
    """
    subject_data = subject.T  # original: (ROI, TR) -> (TR, ROI)
    y_train, y_test = subject_data[train_idx], subject_data[test_idx]

    weights = np.asarray(
        encoding_helpers.get_banded_weights(
            all_features, train_features, test_features, y_train, y_test
        )
    )
    if weights.ndim != 2 or weights.shape != (
        all_train.shape[1], subject_data.shape[1]
    ):
        raise ValueError(
            f"Unexpected weight shape {weights.shape}; expected "
            f"({all_train.shape[1]}, {subject_data.shape[1]}). "
            "Check the original helper's weight-axis convention."
        )

    predicted_full = encoding_helpers.get_predictions_from_weights(all_test, weights)
    scores: dict[str, np.ndarray] = {
        "full": np.asarray(
            encoding_helpers.get_prediction_scores(predicted_full, y_test)
        )
    }
    for name in FEATURE_NAMES:
        subset = group_slices[name]
        partial_prediction = encoding_helpers.get_predictions_from_weights(
            all_test[:, subset], weights[subset]
        )
        scores[name] = np.asarray(
            encoding_helpers.get_prediction_scores(partial_prediction, y_test)
        )
    for name, score in scores.items():
        if score.shape != (subject_data.shape[1],):
            raise ValueError(
                f"Unexpected {name} score shape {score.shape}; expected "
                f"({subject_data.shape[1]},). Verify encoding_helpers.py."
            )
    return weights, scores


def save_scores_and_weights(
    results: list[tuple[np.ndarray, dict[str, np.ndarray]]],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Save the original filenames and return subject-by-ROI score matrices."""
    all_weights = np.stack([weights for weights, _ in results], axis=0)
    np.save(output_dir / "encoding_results_weights.npy", all_weights)
    scores = {
        name: np.stack([item[1][name] for item in results], axis=0)
        for name in ("full",) + FEATURE_NAMES
    }
    for name, filename in SCORE_FILENAMES.items():
        np.save(output_dir / filename, scores[name])
        print(f"Saved {filename}: shape={scores[name].shape}")
    print(f"Saved encoding_results_weights.npy: shape={all_weights.shape}")
    return scores


def run_original_statistical_helpers(
    scores: dict[str, np.ndarray],
    output_dir: Path,
    fdr_helpers: Any,
    n_permutations: int,
    alpha: float,
) -> None:
    """Apply the ORIGINAL null-generation, p-value and FDR helper functions.

    This does not replace helper-defined null statistics, correction scope,
    or significance masking with an assumed alternative.
    """
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    for name, observed in scores.items():
        prefix = NULL_PREFIXES[name]
        null_dist = fdr_helpers.generate_null(
            observed, n_permutations=n_permutations
        )
        null_filename = f"null_{prefix}_feature_dist{n_permutations}.npy"
        if name == "full":
            null_filename = f"null_all_feature_dist{n_permutations}.npy"
        np.save(output_dir / null_filename, null_dist)
        print(f"Saved {null_filename}: shape={np.shape(null_dist)}")

        p_values = np.asarray(fdr_helpers.get_p_values(observed, null_dist))
        if p_values.shape != observed.shape:
            raise ValueError(
                f"Unexpected p-value shape for {name}: {p_values.shape}; "
                f"expected {observed.shape}."
            )
        result_name = RESULT_PREFIXES[name]
        np.save(
            result_dir / f"thresholded_{result_name}_p_values.npy", p_values
        )
        thresholded, n_significant = fdr_helpers.get_fdr_controlled(
            p_values, threshold=alpha, observed=observed
        )
        if np.shape(thresholded) != observed.shape:
            raise ValueError(
                f"Unexpected thresholded score shape for {name}: "
                f"{np.shape(thresholded)}; expected {observed.shape}."
            )
        alpha_suffix = f"{alpha:.10f}".rstrip("0").rstrip(".").split(".")[-1]
        # At alpha=0.05, this is "05" (matching the original filenames).
        np.save(
            result_dir
            / f"encoding_results_{result_name}_scores_thresholded{alpha_suffix}.npy",
            thresholded,
        )
        print(f"{name}: {n_significant} significant entries "
              f"according to the supplied FDR helper (alpha={alpha:g}).")


def main() -> None:
    args = parse_args()
    encoding_helpers = import_project_helper(
        "encoding_helpers", args.encoding_helpers_dir
    )
    fdr_helpers = import_project_helper(
        "fdr_correction_helpers", args.fdr_helpers_dir
    )
    for attribute in ("get_banded_weights", "get_predictions_from_weights",
                      "get_prediction_scores"):
        if not callable(getattr(encoding_helpers, attribute, None)):
            raise AttributeError(f"encoding_helpers.py lacks {attribute}().")
    for attribute in ("generate_null", "get_p_values", "get_fdr_controlled"):
        if not callable(getattr(fdr_helpers, attribute, None)):
            raise AttributeError(f"fdr_correction_helpers.py lacks {attribute}().")

    bold = np.load(args.bold, mmap_mode="r", allow_pickle=False)
    if bold.ndim != 3 or min(bold.shape) == 0:
        raise ValueError("BOLD must have shape (subjects, cortical ROIs, TRs).")
    n_subjects, n_rois, n_trs = bold.shape
    if not np.isfinite(bold).all():
        raise ValueError("BOLD contains NaN/inf; inspect preprocessing first.")

    kernel = spm_style_hrf(
        args.tr, args.hrf_oversampling, args.hrf_duration
    )
    groups, group_slices, full_design = load_design(
        args.feature_dir, args.domain, args.knowledge_prefix, n_trs, kernel
    )
    indices = np.arange(n_trs)
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.split_seed,
        shuffle=True,
    )
    all_train, all_test = full_design[train_idx], full_design[test_idx]
    all_features = list(groups.values())
    train_features = [array[train_idx] for array in all_features]
    test_features = [array[test_idx] for array in all_features]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "bold_shape": [int(n_subjects), int(n_rois), int(n_trs)],
        "subject_order": "Must match the downstream hippocampal activity file.",
        "feature_order": list(FEATURE_NAMES),
        "feature_slices_zero_based_stop_exclusive": {
            key: [value.start, value.stop]
            for key, value in group_slices.items()
        },
        "tr_seconds": args.tr,
        "hrf_oversampling": args.hrf_oversampling,
        "hrf_duration_seconds": args.hrf_duration,
        "test_size": args.test_size,
        "split_seed": args.split_seed,
        "n_permutations": args.n_permutations,
        "fdr_alpha": args.fdr_alpha,
        "preprocessing": "HRF convolution then whole-timeline feature z-scoring",
        "split": "Random TR-level train/test split (not independent run split)",
        "group_scores": "Partial predictions using slices of full-model weights",
        "helper_implementations": "User-supplied original project helpers",
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez(
        args.output_dir / "train_test_indices.npz",
        train_idx=train_idx, test_idx=test_idx,
    )
    print(
        f"Loaded BOLD {bold.shape}; design {full_design.shape}; "
        f"train/test TRs: {len(train_idx)}/{len(test_idx)}"
    )

    results = Parallel(
        n_jobs=args.n_jobs, backend="loky", prefer="processes", batch_size=1
    )(
        delayed(process_one_subject)(
            subject, all_test, all_train, all_features, train_features,
            test_features, train_idx, test_idx, group_slices, encoding_helpers
        )
        for subject in bold
    )
    scores = save_scores_and_weights(results, args.output_dir)
    if args.null_seed is not None:
        # Only controls helper implementations using np.random's legacy RNG.
        np.random.seed(args.null_seed)
    run_original_statistical_helpers(
        scores, args.output_dir, fdr_helpers,
        args.n_permutations, args.fdr_alpha,
    )
    print(f"Analysis completed; outputs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
