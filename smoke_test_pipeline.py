#!/usr/bin/env python3
"""Run SYNTHETIC smoke checks for three analysis scripts.

- Step 2 and Step 3 execute their real supplied analysis code.
- Step 1 "mock" executes the REAL Step 1 wrapper and the original FDR
  function bodies, but replaces tikreg model fitting with ordinary
  least-squares weights solely to check file/array interface compatibility.
- Step 1 "real" executes tikreg and the imported helpers if dependencies exist.

Passing these checks NEVER validates scientific conclusions or reproduces the
original analyses. The 'mock' result is especially unsuitable for inference.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from statsmodels.stats.multitest import multipletests


def execute_subprocess(command: list[str], label: str) -> dict:
    result = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    print(f"\n[{label}] return code: {result.returncode}")
    if result.stdout:
        print(result.stdout[-3500:])
    if result.stderr:
        print("stderr:", result.stderr[-1800:])
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "return_code": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def original_fdr_without_imaging_imports(source_path: Path):
    """Compile unchanged FDR function bodies for a lightweight interface test.

    This only avoids importing optional NIfTI/spatial-analysis packages. It
    does not modify the bodies of generate_null/get_p_values/get_fdr_controlled.
    """
    source = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"rand_sign_flip", "generate_null",
             "get_p_values", "get_fdr_controlled"}
    declarations = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in declarations} == names
    module = ast.Module(body=declarations, type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"np": np, "multipletests": multipletests}
    exec(compile(module, str(source_path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def mock_encoding_helper():
    """Use simple least squares ONLY to verify the Step 1 I/O contract."""
    def get_banded_weights(features, features_train, features_test,
                           bold_train, bold_test):
        X = np.concatenate(features_train, axis=1)
        return np.linalg.lstsq(X, bold_train, rcond=None)[0]

    def get_predictions_from_weights(features, weights):
        return features @ weights

    def get_prediction_scores(predicted, true):
        x = predicted - predicted.mean(axis=0)
        y = true - true.mean(axis=0)
        numerator = np.sum(x * y, axis=0)
        denominator = np.sqrt(np.sum(x*x, axis=0) * np.sum(y*y, axis=0))
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    return SimpleNamespace(
        get_banded_weights=get_banded_weights,
        get_predictions_from_weights=get_predictions_from_weights,
        get_prediction_scores=get_prediction_scores,
    )


def run_step1_mock(project: Path, data: Path, out: Path,
                   permutations: int) -> dict:
    model_path = (
        project / "01_cortical_encoding_model_fixed.py"
        if (project / "01_cortical_encoding_model_fixed.py").is_file()
        else project / "01_cortical_encoding_model.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "step1_mock_interface_only", model_path
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)

    fake_model = mock_encoding_helper()
    original_fdr = original_fdr_without_imaging_imports(
        project / "fdr_correction_helpers.py"
    )
    module.import_project_helper = (
        lambda name, folder: fake_model if name == "encoding_helpers"
        else original_fdr if name == "fdr_correction_helpers"
        else None
    )
    argv = [
        str(model_path),
        "--bold", str(data / "step1_cortical" / "cortical_bold.npy"),
        "--feature-dir", str(data / "step1_cortical" / "features"),
        "--output-dir", str(out / "step1_mock"),
        "--domain", "all",
        "--knowledge-prefix", "know",
        "--n-permutations", str(permutations),
        "--n-jobs", "1",
        "--null-seed", "2026",
    ]
    previous_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = previous_argv

    step_out = out / "step1_mock"
    knowledge_scores = np.load(
        step_out / "encoding_results_f5_know_scores_unthreshold.npy"
    )
    null = np.load(step_out / f"null_f5_know_feature_dist{permutations}.npy")
    p = np.load(
        step_out / "result" / "thresholded_f5_know_p_values.npy"
    )
    thresholded = np.load(
        step_out / "result" / "encoding_results_f5_know_scores_thresholded05.npy"
    )
    manifest = json.loads(
        (data / "synthetic_manifest.json").read_text(encoding="utf-8")
    )
    n_subjects = manifest["n_subjects"]
    n_parcels = manifest["n_cortical_parcels"]
    assert knowledge_scores.shape == (n_subjects, n_parcels)
    assert null.shape == (permutations, n_parcels)
    assert p.shape == (n_parcels,)
    assert thresholded.shape == (n_parcels,)
    assert np.all((p >= 0) & (p <= 1))
    return {
        "status": "PASS",
        "test_type": "MOCK tikreg fitting; real wrapper and original FDR function bodies",
        "knowledge_scores_shape": list(knowledge_scores.shape),
        "null_shape": list(null.shape),
        "parcel_p_shape": list(p.shape),
        "warning": "The fitted model was replaced by least squares: not a real encoding-model test.",
    }


def run_all(project: Path, data: Path, output: Path,
            permutations: int, step1: str):
    project = project.resolve()
    data = data.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (data / "synthetic_manifest.json").read_text(encoding="utf-8")
    )
    results = {
        "notice": "SYNTHETIC interface/execution tests only. Not scientific validation.",
        "manifest": manifest,
        "step1_mode": step1,
        "steps": {},
    }

    # Real Step 2 run
    results["steps"]["step2_real"] = execute_subprocess([
        sys.executable, str(project / "02_extract_hippocampal_activity.py"),
        "--timecourses",
        str(data / "step2_hippocampus" / "hippocampal_timecourses.npy"),
        "--events", str(data / "step2_hippocampus" / "knowledge_events.npy"),
        "--output-dir", str(output / "step2"),
        "--left-index", "8", "--right-index", "9",
        "--lag", "3", "--n-pre", "3", "--n-post", "3",
    ], "Step 2 (real)")

    if results["steps"]["step2_real"]["status"] == "PASS":
        hippo_matrix = np.load(output / "step2" / "hippo_L_subject_time.npy")
        expected = (manifest["n_subjects"], 7)
        assert hippo_matrix.shape == expected
        assert np.isfinite(hippo_matrix).all()
        results["steps"]["step2_real"]["checked_shape"] = list(hippo_matrix.shape)

    # Real Step 3 using its own standalone toy inputs
    results["steps"]["step3_real_standalone"] = execute_subprocess([
        sys.executable, str(project / "03_predict_hippocampal_activity.py"),
        "--encoding-scores",
        str(data / "step3_prediction" / "encoding_scores.npy"),
        "--ifg-indices", str(data / "step3_prediction" / "ifg_indices.npy"),
        "--itg-indices", str(data / "step3_prediction" / "itg_indices.npy"),
        "--mfg-indices", str(data / "step3_prediction" / "mfg_indices.npy"),
        "--hippocampus",
        str(data / "step3_prediction" / "hippo_L_subject_time.npy"),
        "--timepoints",
        str(data / "step3_prediction" / "timepoints_order.npy"),
        "--output-dir", str(output / "step3_standalone"),
        "--n-permutations", str(permutations), "--seed", "2026",
    ], "Step 3 (real, standalone data)")

    if results["steps"]["step3_real_standalone"]["status"] == "PASS":
        import pandas as pd
        csv_path = output / "step3_standalone" / (
            "SVR_cortical_encoding_predict_hippo_L_time_resolved_results.csv"
        )
        table = pd.read_csv(csv_path)
        assert len(table) == 7 and table["p_perm_fdr"].notna().any()
        null = np.load(
            output / "step3_standalone"
            / "SVR_null_distribution_timepoint_by_perm.npy"
        )
        assert null.shape == (7, permutations)
        results["steps"]["step3_real_standalone"]["null_shape"] = list(null.shape)

    if step1 == "mock":
        try:
            results["steps"]["step1_mock_interface"] = run_step1_mock(
                project, data, output, permutations
            )
        except Exception as error:
            results["steps"]["step1_mock_interface"] = {
                "status": "FAIL",
                "error": str(error),
                "traceback": traceback.format_exc()[-2500:],
            }
            print("\n[Step 1 mock] FAIL:", error)
    elif step1 == "real":
        model_path = (
            project / "01_cortical_encoding_model_fixed.py"
            if (project / "01_cortical_encoding_model_fixed.py").exists()
            else project / "01_cortical_encoding_model.py"
        )
        results["steps"]["step1_real"] = execute_subprocess([
            sys.executable, str(model_path),
            "--bold",
            str(data / "step1_cortical" / "cortical_bold.npy"),
            "--feature-dir", str(data / "step1_cortical" / "features"),
            "--output-dir", str(output / "step1_real"),
            "--domain", "all", "--knowledge-prefix", "know",
            "--n-permutations", str(permutations),
            "--n-jobs", "1", "--null-seed", "2026",
        ], "Step 1 (real tikreg and real FDR imports)")
    else:
        results["steps"]["step1"] = {
            "status": "SKIP", "reason": "Requested --step1 skip"
        }

    # Also verify that Step 2 output can be fed to the REAL Step 3 script.
    if results["steps"]["step2_real"]["status"] == "PASS":
        results["steps"]["step2_to_step3_real"] = execute_subprocess([
            sys.executable, str(project / "03_predict_hippocampal_activity.py"),
            "--encoding-scores",
            str(data / "step3_prediction" / "encoding_scores.npy"),
            "--ifg-indices", str(data / "step3_prediction" / "ifg_indices.npy"),
            "--itg-indices", str(data / "step3_prediction" / "itg_indices.npy"),
            "--mfg-indices", str(data / "step3_prediction" / "mfg_indices.npy"),
            "--hippocampus", str(output / "step2" / "hippo_L_subject_time.npy"),
            "--timepoints", str(output / "step2" / "timepoints_order.npy"),
            "--output-dir", str(output / "step3_chained"),
            "--n-permutations", str(permutations), "--seed", "2026",
        ], "Step 2 -> Step 3 (real scripts chained)")

    report_path = output / "SMOKE_TEST_REPORT.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nReport saved to:", report_path)
    for name, step in results["steps"].items():
        print(f"  {name}: {step['status']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--step1", choices=["mock", "real", "skip"],
                        default="mock")
    args = parser.parse_args()
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive.")
    project = args.project_dir.resolve()
    data = args.data_dir or project / "simulated_data"
    out = args.output_dir or project / "results_smoke"
    results = run_all(project, data, out, args.n_permutations, args.step1)
    if any(result["status"] == "FAIL" for result in results["steps"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
