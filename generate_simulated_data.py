#!/usr/bin/env python3
"""Generate small, reproducible SYNTHETIC inputs for the three analysis scripts.

The arrays match the scripts' stated input shapes and file naming conventions.
They are not derived from human participants and are NOT suitable for scientific
inference or reproducing the findings of the accompanying manuscript.
"""

from __future__ import annotations

import argparse
import json
from math import gamma
from pathlib import Path

import numpy as np


def hrf_kernel(tr: float = 2.0, oversampling: int = 16, duration: float = 32.0):
    """Match the double-gamma-style sampling used in Step 1."""
    fine_time = np.arange(0, duration, tr / oversampling)
    def gamma_pdf(x, shape):
        return x ** (shape - 1) * np.exp(-x) / gamma(shape)
    response = gamma_pdf(fine_time, 6.0) - gamma_pdf(fine_time, 16.0) / 6.0
    response /= response.sum()
    return response[::oversampling]


def convolve_and_standardize(features: np.ndarray, kernel: np.ndarray):
    x = np.column_stack([
        np.convolve(features[:, col], kernel, mode="full")[:features.shape[0]]
        for col in range(features.shape[1])
    ])
    std = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(std > 0, std, 1)


def smooth_features(rng: np.random.Generator, n_trs: int, n_features: int):
    white_noise = rng.normal(size=(n_trs, n_features))
    smooth = np.empty_like(white_noise)
    for col in range(n_features):
        smooth[:, col] = np.convolve(
            white_noise[:, col], [0.2, 0.6, 0.2], mode="same"
        )
    return smooth


def generate(output_dir: Path, seed: int, n_subjects: int,
             n_parcels: int, n_trs: int):
    if n_subjects < 6 or n_parcels < 12 or n_trs < 96:
        raise ValueError(
            "Use at least 6 subjects, 12 parcels, and 96 TRs for this example."
        )
    rng = np.random.default_rng(seed)
    output_dir = output_dir.resolve()

    step1 = output_dir / "step1_cortical"
    feature_dir = step1 / "features"
    step2 = output_dir / "step2_hippocampus"
    step3 = output_dir / "step3_prediction"
    for folder in [step1, feature_dir, step2, step3]:
        folder.mkdir(parents=True, exist_ok=True)

    # Short multi-TR event blocks are spaced far enough apart to keep
    # complete three-TR pre/post windows after the three-TR HRF shift.
    knowledge_events = np.zeros(n_trs, dtype=np.int8)
    event_starts = list(range(12, n_trs - 12, 16))
    for start in event_starts:
        knowledge_events[start:start + 2] = 1

    features = {
        "vis": smooth_features(rng, n_trs, 3),
        "aud": smooth_features(rng, n_trs, 2),
        "cha": smooth_features(rng, n_trs, 2),
        "sem": smooth_features(rng, n_trs, 4),
        "know": knowledge_events.astype(float)[:, None],
    }
    feature_filenames = {
        "vis": "vis_all.npy",
        "aud": "aud_all.npy",
        "cha": "cha_all_new.npy",
        "sem": "sem_all_word2vec.npy",
        "know": "know_all.npy",
    }
    for feature, filename in feature_filenames.items():
        np.save(feature_dir / filename, features[feature])

    # The five groups are convolved before generating cortical responses.
    kernel = hrf_kernel()
    design = np.concatenate([
        convolve_and_standardize(features[name], kernel)
        for name in ("vis", "aud", "cha", "sem", "know")
    ], axis=1)
    n_features = design.shape[1]

    # Parcel groups serve only as example ROI index files:
    # IFG = 0..3, ITG = 4..7, MFG = 8..11.
    roi_indices = {
        "ifg": np.arange(0, 4, dtype=np.int64),
        "itg": np.arange(4, 8, dtype=np.int64),
        "mfg": np.arange(8, 12, dtype=np.int64),
    }
    for name, indices in roi_indices.items():
        np.save(step3 / f"{name}_indices.npy", indices)

    subject_traits = rng.normal(size=(n_subjects, 3))
    base_weights = rng.normal(0, 0.25, size=(n_features, n_parcels))
    # Last feature column is the knowledge-event vector.
    base_weights[-1, :] = 0.06
    base_weights[-1, :12] = 0.75

    bold = np.empty((n_subjects, n_parcels, n_trs), dtype=np.float32)
    for s in range(n_subjects):
        individual_weights = base_weights + rng.normal(
            0, 0.06, size=base_weights.shape
        )
        for k, indices in enumerate(roi_indices.values()):
            individual_weights[-1, indices] += 0.12 * subject_traits[s, k]
        simulated = design @ individual_weights
        simulated += rng.normal(0, 0.7, size=simulated.shape)
        bold[s] = simulated.T.astype(np.float32)

    np.save(step1 / "cortical_bold.npy", bold)

    # Ten illustrative hippocampal/other ROI channels. The script uses
    # channels 8 and 9 as left/right hippocampus by default.
    hippo = rng.normal(
        0, 0.55, size=(10, n_subjects, n_trs)
    ).astype(np.float32)
    shifted_event = np.zeros(n_trs, dtype=float)
    shifted_event[3:] = knowledge_events[:-3]  # HRF lag = 3 TRs
    event_response = np.convolve(
        shifted_event, np.array([0.25, 0.55, 0.20]), mode="full"
    )[:n_trs]
    for s in range(n_subjects):
        l_amp = max(0.15, 0.85 + 0.2 * subject_traits[s, 0])
        r_amp = max(0.15, 0.70 + 0.2 * subject_traits[s, 1])
        hippo[8, s] += (l_amp * event_response).astype(np.float32)
        hippo[9, s] += (r_amp * event_response).astype(np.float32)
    np.save(step2 / "hippocampal_timecourses.npy", hippo)
    np.save(step2 / "knowledge_events.npy", knowledge_events)

    # Standalone inputs for Step 3. These do not depend on completing the
    # computationally heavier tikreg model in Step 1.
    scores = rng.normal(
        0.13, 0.025, size=(n_subjects, n_parcels)
    ).astype(np.float64)
    for k, indices in enumerate(roi_indices.values()):
        scores[:, indices] = (
            0.31
            + 0.13 * subject_traits[:, k, None]
            + rng.normal(0, 0.025, size=(n_subjects, len(indices)))
        )
    scores = np.clip(scores, -0.6, 0.9)
    np.save(step3 / "encoding_scores.npy", scores)

    labels = ["Pre-3", "Pre-2", "Pre-1", "During",
              "Post+1", "Post+2", "Post+3"]
    np.save(step3 / "timepoints_order.npy", np.asarray(labels))
    means = np.column_stack([
        scores[:, roi_indices[key]].mean(axis=1)
        for key in ("ifg", "itg", "mfg")
    ])
    y = np.empty((n_subjects, len(labels)), dtype=float)
    varying_coefficients = [0.1, 0.2, 0.25, 1.0, 0.6, 0.35, 0.15]
    for j, scale in enumerate(varying_coefficients):
        y[:, j] = (
            scale * (1.6 * means[:, 0] - 1.1 * means[:, 1]
                     + 0.75 * means[:, 2])
            + rng.normal(0, 0.045, size=n_subjects)
            + 0.02 * j
        )
    np.save(step3 / "hippo_L_subject_time.npy", y)

    manifest = {
        "warning": "ALL ARRAYS ARE SYNTHETIC: use only to check code execution.",
        "random_seed": seed,
        "n_subjects": n_subjects,
        "n_cortical_parcels": n_parcels,
        "n_TRs": n_trs,
        "TR_seconds": 2.0,
        "knowledge_event_blocks": len(event_starts),
        "knowledge_event_TRs": int(knowledge_events.sum()),
        "step1": {
            "bold_shape": list(bold.shape),
            "design_shape": list(design.shape),
            "feature_files": {
                feature_filenames[name]: list(features[name].shape)
                for name in feature_filenames
            },
        },
        "step2": {
            "hippocampal_timecourses_shape": list(hippo.shape),
            "knowledge_events_shape": list(knowledge_events.shape),
            "hippocampus_channels_zero_based": [8, 9],
        },
        "step3": {
            "encoding_scores_shape": list(scores.shape),
            "hippocampus_target_shape": list(y.shape),
            "timepoints": labels,
            "roi_indices_zero_based": {
                key: val.tolist() for key, val in roi_indices.items()
            },
        },
        "note": "Subject indices refer to the same synthetic participants in each step.",
    }
    (output_dir / "synthetic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Synthetic data generated at:", output_dir)
    print("Step 1 cortical BOLD:", bold.shape, "| full design:", design.shape)
    print("Step 2 hippocampal time series:", hippo.shape,
          "| event blocks:", len(event_starts))
    print("Step 3 encoding scores:", scores.shape,
          "| hippocampal target:", y.shape)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("simulated_data"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--n-parcels", type=int, default=18)
    parser.add_argument("--n-trs", type=int, default=160)
    args = parser.parse_args()
    generate(args.output_dir, args.seed, args.n_subjects,
             args.n_parcels, args.n_trs)


if __name__ == "__main__":
    main()
