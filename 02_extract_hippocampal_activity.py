#!/usr/bin/env python3
"""Extract peri-knowledge-event hippocampal BOLD activity.

Input time courses must have shape (n_rois, n_subjects, n_trs).
The event vector must contain one binary (0/1) label per TR.
ROI indices and subject ordering must be verified against the source data.

"During" is the average z-scored BOLD activity over each event after
shifting the event boundaries by the specified HRF lag. Pre-k and
Post+k correspond to single TRs relative to those shifted boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_event_blocks(events: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, stop) pairs for runs of 1s; stop is exclusive."""
    padded = np.pad(events.astype(int), (1, 1), constant_values=0)
    differences = np.diff(padded)
    starts = np.flatnonzero(differences == 1)
    stops = np.flatnonzero(differences == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def standardize_timecourse(values: np.ndarray) -> np.ndarray:
    """Compute within-subject temporal z-scores, ignoring missing values."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if finite.sum() < 2:
        return result
    mean = values[finite].mean()
    std = values[finite].std(ddof=0)
    if std > 0:
        result[finite] = (values[finite] - mean) / std
    return result


def extract_subject_matrix(
    timecourses: np.ndarray,
    events: np.ndarray,
    roi_index: int,
    lag: int,
    n_pre: int,
    n_post: int,
    strict_neighbors: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Average valid events within each subject and relative-time bin.

    Events without the complete stimulus/BOLD extraction window are
    excluded. When strict_neighbors=True, a pre/post bin is excluded
    if its corresponding stimulus TR belongs to another event.

    Returns
    -------
    means, counts : arrays of shape (n_subjects, n_time_bins)
        Means and numbers of contributing events for each bin.
    labels : list[str]
        Ordered relative-time-bin names.
    n_eligible_events : int
        Number of events with complete extraction windows.
    """
    _, n_subjects, n_trs = timecourses.shape
    if not 0 <= roi_index < timecourses.shape[0]:
        raise ValueError(f"ROI index {roi_index} is outside the available ROIs.")

    labels = (
        [f"Pre-{i}" for i in range(n_pre, 0, -1)]
        + ["During"]
        + [f"Post+{i}" for i in range(1, n_post + 1)]
    )
    blocks = find_event_blocks(events)
    eligible = [
        (start, stop)
        for start, stop in blocks
        if start - n_pre >= 0
        and stop + n_post <= n_trs
        and start + lag - n_pre >= 0
        and stop + lag + n_post <= n_trs
    ]
    if not eligible:
        raise ValueError("No events have a complete pre/during/post window.")

    means = np.full((n_subjects, len(labels)), np.nan)
    counts = np.zeros((n_subjects, len(labels)), dtype=int)
    label_index = {label: index for index, label in enumerate(labels)}

    for subject in range(n_subjects):
        signal = standardize_timecourse(timecourses[roi_index, subject])
        collected = {label: [] for label in labels}

        for start, stop in eligible:
            bold_start, bold_stop = start + lag, stop + lag

            during = signal[bold_start:bold_stop]
            finite_during = during[np.isfinite(during)]
            if finite_during.size:
                collected["During"].append(float(finite_during.mean()))

            for i in range(1, n_pre + 1):
                stimulus_index = start - i
                bold_index = bold_start - i
                if strict_neighbors and events[stimulus_index] == 1:
                    continue
                if np.isfinite(signal[bold_index]):
                    collected[f"Pre-{i}"].append(float(signal[bold_index]))

            for i in range(1, n_post + 1):
                # stop is exclusive: Post+1 starts at stop + lag.
                stimulus_index = stop + i - 1
                bold_index = bold_stop + i - 1
                if strict_neighbors and events[stimulus_index] == 1:
                    continue
                if np.isfinite(signal[bold_index]):
                    collected[f"Post+{i}"].append(float(signal[bold_index]))

        for label, values in collected.items():
            column = label_index[label]
            counts[subject, column] = len(values)
            if values:
                means[subject, column] = np.mean(values)

    return means, counts, labels, len(eligible)


def save_long_format(
    means: np.ndarray, counts: np.ndarray, labels: list[str], path: Path
) -> None:
    """Save subject-level bin averages, retaining event counts."""
    records = [
        {
            "Subject": subject,
            "Timepoint": label,
            "BOLD_Z": means[subject, column],
            "N_events": counts[subject, column],
        }
        for subject in range(means.shape[0])
        for column, label in enumerate(labels)
    ]
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def plot_activity(
    means: np.ndarray,
    labels: list[str],
    title: str,
    color: str,
    output_path: Path,
) -> None:
    """Plot group means and between-subject SEM for each time bin."""
    x = np.arange(len(labels))
    mean_values = np.full(len(labels), np.nan)
    sem_values = np.full(len(labels), np.nan)
    for column in range(len(labels)):
        valid = means[:, column]
        valid = valid[np.isfinite(valid)]
        if len(valid):
            mean_values[column] = valid.mean()
        if len(valid) > 1:
            sem_values[column] = valid.std(ddof=1) / np.sqrt(len(valid))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, mean_values, yerr=sem_values, color=color,
                marker="o", linewidth=2, capsize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvspan(labels.index("During") - 0.4, labels.index("During") + 0.4,
               color="gray", alpha=0.15)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_xlabel("Relative time")
    ax.set_ylabel("Mean BOLD signal (within-subject z-score)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timecourses", type=Path, required=True,
                        help="ROI x subject x TR .npy time-course array")
    parser.add_argument("--events", type=Path, required=True,
                        help="Binary knowledge-event vector (.npy)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-index", type=int, default=8,
                        help="Zero-based left hippocampus ROI index; verify mapping")
    parser.add_argument("--right-index", type=int, default=9,
                        help="Zero-based right hippocampus ROI index; verify mapping")
    parser.add_argument("--lag", type=int, default=3, help="HRF shift in TRs")
    parser.add_argument("--n-pre", type=int, default=3)
    parser.add_argument("--n-post", type=int, default=3)
    args = parser.parse_args()

    if args.lag < 0 or args.n_pre < 0 or args.n_post < 0:
        parser.error("--lag, --n-pre, and --n-post must be non-negative.")
    if args.left_index == args.right_index:
        parser.error("Left and right hippocampal ROI indices must differ.")

    timecourses = np.load(args.timecourses, allow_pickle=False)
    events = np.asarray(np.load(args.events, allow_pickle=False)).squeeze()
    if timecourses.ndim != 3:
        raise ValueError("Time courses must have shape (ROI, subject, TR).")
    if events.ndim != 1 or events.shape[0] != timecourses.shape[2]:
        raise ValueError("Event vector must be 1D and match the TR dimension.")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("Event vector must contain only 0 and 1.")
    if not 0 <= args.left_index < timecourses.shape[0]:
        raise ValueError("Left ROI index is out of bounds.")
    if not 0 <= args.right_index < timecourses.shape[0]:
        raise ValueError("Right ROI index is out of bounds.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    left, left_counts, labels, n_events = extract_subject_matrix(
        timecourses, events, args.left_index, args.lag, args.n_pre, args.n_post
    )
    right, right_counts, _, _ = extract_subject_matrix(
        timecourses, events, args.right_index, args.lag, args.n_pre, args.n_post
    )
    pair = np.stack([left, right], axis=1)
    valid_count = np.isfinite(pair).sum(axis=1)
    bilateral = np.divide(
        np.nansum(pair, axis=1), valid_count,
        out=np.full(left.shape, np.nan), where=valid_count > 0
    )

    for name, values in [
        ("hippo_L_subject_time", left),
        ("hippo_R_subject_time", right),
        ("hippo_bilateral_subject_time", bilateral),
    ]:
        np.save(args.output_dir / f"{name}.npy", values)
    np.save(args.output_dir / "timepoints_order.npy", np.asarray(labels))
    save_long_format(
        left, left_counts, labels,
        args.output_dir / "hippo_L_all_events_long.csv"
    )
    save_long_format(
        right, right_counts, labels,
        args.output_dir / "hippo_R_all_events_long.csv"
    )
    plot_activity(
        left, labels, "Left hippocampal peri-knowledge activity", "lightcoral",
        args.output_dir / "zzzzhippo_L_all_events_activity_curve.png"
    )
    plot_activity(
        right, labels, "Right hippocampal peri-knowledge activity", "firebrick",
        args.output_dir / "zzzzhippo_R_all_events_activity_curve.png"
    )
    print(f"Input: {timecourses.shape[1]} subjects; {len(find_event_blocks(events))} "
          f"events; {n_events} events with complete windows.")
    print(f"Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
