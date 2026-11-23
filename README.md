# Cortical Encoding of Knowledge Events and Hippocampal Dynamics during Naturalistic Learning in Children

Research scripts supporting analyses of cortical encoding of knowledge events and hippocampal activity during naturalistic educational video viewing.

This repository contains a Python workflow for (1) fitting cortical encoding models, (2) extracting peri-knowledge-event hippocampal BOLD activity, and (3) predicting hippocampal activity from cortical encoding scores. It also includes helper functions and, where provided, related MATLAB scripts and a legacy Jupyter notebook.

> **Data availability:** Participant-level research data are not distributed here. The `simulated_data/` directory contains **artificial test arrays only**, not observed brain or behavioral data. Synthetic tests check interfaces and code execution, **not scientific validity or reproduction of the manuscript results**.

## 1. Repository files

| File | Purpose |
| --- | --- |
| `01_cortical_encoding_model.py` | Fit a five-feature cortical encoding model for each participant and compute full-model and feature-specific prediction scores. |
| `02_extract_hippocampal_activity.py` | Extract and visualize hippocampal BOLD activity before, during, and after knowledge events. |
| `03_predict_hippocampal_activity.py` | Use IFG, ITG, and MFG cortical encoding scores to predict hippocampal activity via leave-one-subject-out (LOSO) support vector regression. |
| `encoding_helpers.py` | Model-fitting, kernel-to-primal weight conversion, and prediction-scoring utilities used by the cortical model. |
| `fdr_correction_helpers.py` | Permutation, p-value, and false discovery rate (FDR) utilities. |
| `data_prep_helpers_new.py` | NIfTI masking, data preparation, and ISC utilities; imported by the original FDR helper. |
| `generate_simulated_data.py` | Reproducibly generate example `.npy` inputs for all three workflows. |
| `smoke_test_pipeline.py` | Run synthetic-data checks and save a JSON pass/fail report. |
| `simulated_data/` | Small example arrays for Steps 1–3; **not real participant data**. |

**Other analysis files, if included:** `encoding_model_02EXTENDED-Copy1.ipynb` is an earlier notebook-based cortical encoding workflow. `generate_regressor.m`, `rating_jaccard_check.m`, `rating_plot.m`, and `rating_transform.m` concern knowledge-event annotation, event regressors, and agreement; `voxelwise_encoding_cluster_corr_new.m` and `voxelwise_encoding_group_analysis_npc.m` contain separate voxelwise MATLAB analyses. These MATLAB workflows are not invoked automatically by the three Python scripts.

The legacy notebook contains machine-specific absolute paths. The command-line Python scripts provide a more portable interface using command-line arguments.

## 2. Software requirements

The Python scripts use Python **3.10 or later** syntax. Required third-party packages, based on the imports in the supplied source files, are:

- `numpy`, `scipy`, `pandas`, `matplotlib`
- `scikit-learn`, `joblib`, `statsmodels`
- `tikreg` (banded ridge encoding models)
- `nibabel`, `nilearn`, `brainiak` (used by the supplied imaging/data-preparation helpers)

Create a Python environment and install the dependencies supported by your platform. For example:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install numpy scipy pandas matplotlib scikit-learn joblib statsmodels
# For real Step 1, also install tikreg, nibabel, nilearn, and brainiak
# following the official instructions for your platform.
```

**Local dependency now included:** `data_prep_helpers_new.py` is supplied alongside `fdr_correction_helpers.py`. Place both in the same repository directory or ensure that directory is on `PYTHONPATH`. Its module-level imports require `nibabel`, `nilearn`, and `brainiak` even when only non-imaging permutation routines are used. The smoke test's **mock** mode bypasses these imaging imports when testing the original FDR function bodies; this is not a substitute for the real dependency checks.

If using the MATLAB scripts, MATLAB and the original workflow's relevant toolboxes are also required (for example, SPM for `spm_hrf`, an appropriate NIfTI I/O toolbox for `load_untouch_nii`, and MATLAB statistics/parallel functionality where used). The required versions were not specified in the supplied files.

## 3. Input files and array conventions

### 3.1 Cortical BOLD time series

Supply a NumPy `.npy` file to `01_cortical_encoding_model.py`:

```text
cortical_bold.npy                 (n_subjects, n_cortical_parcels, n_TRs)
```

The number of TRs must match **every** feature matrix. All participant-level arrays used across the different scripts must use the same participant ordering.

### 3.2 Five cortical encoding features

The cortical model expects the following files within a single `--feature-dir`. With `--domain all` and `--knowledge-prefix know`, filenames are:

```text
features/
├── vis_all.npy
├── aud_all.npy
├── cha_all_new.npy
├── sem_all_word2vec.npy
└── know_all.npy
```

Each array must have shape `(n_TRs, n_features)`; single-feature vectors of shape `(n_TRs,)` are also accepted. Feature groups are concatenated in this order: `vis`, `aud`, `cha`, `sem`, `know`.

**Naming note:** The default `--knowledge-prefix` in the cortical script is `know_8`, which would look for `know_8_all.npy`. Use `--knowledge-prefix know` if your file is named `know_all.npy`. For domain-specific analyses, set `--domain` to the suffix actually used by your feature filenames.

### 3.3 Hippocampal BOLD time series and event labels

The hippocampal extraction script expects:

```text
hippocampal_timecourses.npy       (n_ROIs, n_subjects, n_TRs)
knowledge_events.npy             (n_TRs,) with values 0 or 1
```

The default left/right hippocampal ROI indices are `8` and `9` **using zero-based indexing**. Check these indices against your atlas before running the script. The default temporal HRF shift is `3` TRs; the extraction window includes three pre-event and three post-event TRs. Consecutive event-labeled TRs are treated as one event block.

### 3.4 Cortical ROI index files for hippocampal prediction

`03_predict_hippocampal_activity.py` requires `.npy` files containing integer cortical parcel indices for IFG, ITG, and MFG:

```text
ifg_indices.npy
itg_indices.npy
mfg_indices.npy
```

Indices are zero-based unless `--one-based-indices` is specified. The script averages each participant's encoding scores across parcels in the corresponding ROI. The ROI definitions and subject ordering must be appropriate for the intended analysis.

## 4. Example workflow

The commands below illustrate the expected interface. They assume you have prepared all inputs and installed the required helper modules. **They are not a report of an executed reproduction.**

### Step 1 — Cortical encoding model

```bash
python 01_cortical_encoding_model.py \
  --bold data/cortical_bold.npy \
  --feature-dir data/features \
  --output-dir results/cortical_encoding \
  --domain all \
  --knowledge-prefix know \
  --tr 2.0 \
  --n-permutations 1000 \
  --n-jobs 4
```

Optionally provide `--encoding-helpers-dir` and `--fdr-helpers-dir` if the corresponding modules are stored in separate directories. Use `python 01_cortical_encoding_model.py --help` for the full set of arguments.

**Main outputs (subject-by-parcel arrays where applicable):**

```text
results/cortical_encoding/
├── encoding_results_weights.npy
├── encoding_results_full_scores_unthreshold.npy
├── encoding_results_f1_vis_scores_unthreshold.npy
├── encoding_results_f2_aud_scores_unthreshold.npy
├── encoding_results_f3_cha_scores_unthreshold.npy
├── encoding_results_f4_sem_scores_unthreshold.npy
├── encoding_results_f5_know_scores_unthreshold.npy
├── train_test_indices.npz
├── analysis_metadata.json
└── result/                       # statistical outputs, if this stage completes
```

The script fits a banded-ridge model using all five feature groups, then evaluates contributions from the fitted weights associated with each feature group. Feature-specific scores are **not** obtained by independently refitting five separate models.

**Important dimensionality fix:** This updated test kit includes a corrected `01_cortical_encoding_model.py` that transposes the permutation null from `(permutations, parcels)` to `(parcels, permutations)` when passing it to `get_p_values()` and checks the expected one-dimensional `(parcels,)` p-values. The original null-generation, one-tailed p-value, and BH-FDR function bodies were **not altered**. The supplied model-fitting procedure still requires `tikreg` and has **not been tested end-to-end in this environment**. If using the individual download `01_cortical_encoding_model_fixed.py`, rename it `01_cortical_encoding_model.py` in your repository.

### Step 2 — Peri-event hippocampal activity

```bash
python 02_extract_hippocampal_activity.py \
  --timecourses data/hippocampal_timecourses.npy \
  --events data/knowledge_events.npy \
  --output-dir results/hippocampus \
  --left-index 8 \
  --right-index 9 \
  --lag 3 \
  --n-pre 3 \
  --n-post 3
```

Key outputs:

```text
results/hippocampus/
├── hippo_L_subject_time.npy
├── hippo_R_subject_time.npy
├── hippo_bilateral_subject_time.npy
├── timepoints_order.npy
├── hippo_L_all_events_long.csv
├── hippo_R_all_events_long.csv
├── zzzzhippo_L_all_events_activity_curve.png
└── zzzzhippo_R_all_events_activity_curve.png
```

The saved `.npy` matrices contain mean within-subject z-scored activity for the ordered `Pre-k`, `During`, and `Post+k` bins.

### Step 3 — Predict hippocampal activity from cortical encoding

```bash
python 03_predict_hippocampal_activity.py \
  --encoding-scores results/cortical_encoding/encoding_results_f5_know_scores_unthreshold.npy \
  --ifg-indices data/ifg_indices.npy \
  --itg-indices data/itg_indices.npy \
  --mfg-indices data/mfg_indices.npy \
  --hippocampus results/hippocampus/hippo_L_subject_time.npy \
  --timepoints results/hippocampus/timepoints_order.npy \
  --output-dir results/hippocampal_prediction \
  --n-permutations 1000 \
  --seed 2026
```

This script constructs three cortical features (one each for IFG, ITG, and MFG), fits a linear support vector regression model with leave-one-subject-out cross-validation, and evaluates Pearson correlation between observed and out-of-fold predicted hippocampal activity. It uses target-label permutation testing and BH-FDR correction across the tested relative-time bins.

Key outputs include:

```text
results/hippocampal_prediction/
├── SVR_cortical_features_IFG_ITG_MFG.npy
├── SVR_cortical_encoding_predict_hippo_L_time_resolved_results.csv
├── SVR_null_distribution_timepoint_by_perm.npy
├── SVR_cortical_encoding_predict_hippo_L_time_resolved_r.png
└── prediction_observed_<timepoint>.csv
```

## 5. Analysis and reproducibility notes

- **TR sampling:** The cortical script defaults to `TR = 2 s` and performs HRF convolution and z-scoring across the complete feature timeline before the train/test split.
- **Train/test partition:** The cortical script uses a randomly shuffled TR-level split (default test fraction `0.2`, random state `4`). This is not a held-out-video or held-out-subject evaluation.
- **Permutation and FDR:** The cortical model delegates its null generation and FDR procedures to the supplied `fdr_correction_helpers.py`. Check the helper's output dimensions and statistical assumptions before reporting results. The hippocampal SVR analysis performs its own target-label permutations and time-bin-wise BH-FDR correction.
- **Reproducibility:** The cortical script saves split indices and analysis settings. `--null-seed` only controls helper code that uses NumPy's legacy global random number generator.
- **Participant alignment:** Subject order must match between cortical encoding scores and hippocampal activity. The scripts check array dimensions but cannot verify subject identities.
- **ROI interpretation:** Ensure that the IFG/ITG/MFG parcel index files match the parcel atlas and that the selected regions are appropriate for the statistical question.
- **Environment:** A synthetic test dataset is now included, but this is not a public substitute for the original data. Exact environment reproduction still requires documenting the original package versions and matching research inputs.

## 6. Citation and contact

If this repository is used in published work, please cite the associated manuscript when bibliographic information is available. Repository maintainers can add their preferred contact information and license here.


## 7. Verify the code using synthetic data

This repository includes artificial `.npy` arrays and a self-contained
`generate_simulated_data.py` script. The **default** example is small:

| Item | Shape | Interpretation |
| --- | --- | --- |
| Cortical BOLD (`step1_cortical/cortical_bold.npy`) | `(10, 18, 160)` | 10 simulated subjects, 18 cortical parcels, 160 TRs |
| Visual (`vis_all.npy`) | `(160, 3)` | Three visual feature columns |
| Auditory (`aud_all.npy`) | `(160, 2)` | Two auditory feature columns |
| Character (`cha_all_new.npy`) | `(160, 2)` | Two character-related feature columns |
| Semantic (`sem_all_word2vec.npy`) | `(160, 4)` | Four example semantic feature columns |
| Knowledge (`know_all.npy`) | `(160, 1)` | Binary event label as a model regressor |
| Hippocampal BOLD (`step2_hippocampus/hippocampal_timecourses.npy`) | `(10, 10, 160)` | 10 ROI channels × 10 subjects × 160 TRs; channels 8 and 9 are the example left/right hippocampus |
| Event vector (`step2_hippocampus/knowledge_events.npy`) | `(160,)` | Nine knowledge-event blocks, each lasting two TRs |
| Cortical encoding scores (`step3_prediction/encoding_scores.npy`) | `(10, 18)` | Artificial encoding-score matrix for independent Step 3 testing |
| Step 3 hippocampal activity (`step3_prediction/hippo_L_subject_time.npy`) | `(10, 7)` | Artificial targets for seven peri-event time bins |
| Time labels (`step3_prediction/timepoints_order.npy`) | `(7,)` | `Pre-3`, `Pre-2`, `Pre-1`, `During`, `Post+1`, `Post+2`, `Post+3` |
| ROI index files (`ifg_indices.npy`, `itg_indices.npy`, `mfg_indices.npy`) | `(4,)` each | Zero-based parcel IDs; three non-overlapping example ROIs |

The generator also creates `simulated_data/synthetic_manifest.json`,
which contains the exact shapes, event counts, and ROI indices.

### 7.1 Generate or regenerate all test inputs

From the repository root:

```bash
python generate_simulated_data.py \
  --output-dir simulated_data \
  --seed 2026 \
  --n-subjects 10 \
  --n-parcels 18 \
  --n-trs 160
```

The repository ZIP already includes these generated `.npy` files, so running
the generator again is optional.

### 7.2 One-command smoke test (recommended first)

Install the common Python dependencies shown in Section 2, then run:

```bash
python smoke_test_pipeline.py --step1 mock --n-permutations 10
```

The test script runs:

1. **Real Step 2:** event-locked hippocampal signal extraction and PNG/CSV export.
2. **Real Step 3:** LOSO-SVR predictions and permutation/FDR outputs using
   standalone toy encoding scores and hippocampal targets.
3. **Step 1 interface test:** real Step 1 file loading, HRF convolution,
   subject-level result writing, and the unchanged original
   `generate_null()`, `get_p_values()`, and `get_fdr_controlled()` function
   bodies. **Only `tikreg` model fitting is replaced** with least squares
   in this mock test. This does NOT validate banded-ridge hyperparameter
   selection, estimation, or performance.
4. **Chained real Steps 2 → 3:** feeds the real Step 2 output into Step 3,
   using the standalone toy cortical scores for the other input.

Check `results_smoke/SMOKE_TEST_REPORT.json` for `PASS`/`FAIL` flags and
the checked array shapes. The test returns a nonzero exit code on failure.

**Observed smoke-test status for the supplied examples (synthetic data):**
Steps 2 and 3 and their chained run passed; the Step 1 interface test
also passed with the least-squares mock. **Real `tikreg` fitting was not
executed** because the testing environment lacked `tikreg` and several
neuroimaging packages.

### 7.3 Run each script manually

**Step 1: Real cortical encoding model (requires `tikreg` and all imaging dependencies)**

```bash
python 01_cortical_encoding_model.py \
  --bold simulated_data/step1_cortical/cortical_bold.npy \
  --feature-dir simulated_data/step1_cortical/features \
  --output-dir results_smoke/step1_real \
  --domain all \
  --knowledge-prefix know \
  --tr 2.0 \
  --n-permutations 20 \
  --n-jobs 1 \
  --null-seed 2026
```

Alternatively, use the test runner with the full original-model code path:

```bash
python smoke_test_pipeline.py --step1 real --n-permutations 20
```

This requires all packages imported by `encoding_helpers.py`,
`fdr_correction_helpers.py`, and `data_prep_helpers_new.py`. The
banded-ridge fitting includes cross-validation and may be computationally
expensive; do not interpret this synthetic run as an exact scientific
reproduction.

**Step 2: Extract hippocampal activity**

```bash
python 02_extract_hippocampal_activity.py \
  --timecourses simulated_data/step2_hippocampus/hippocampal_timecourses.npy \
  --events simulated_data/step2_hippocampus/knowledge_events.npy \
  --output-dir results_smoke/hippocampus_manual \
  --left-index 8 \
  --right-index 9 \
  --lag 3 \
  --n-pre 3 \
  --n-post 3
```

Expected core outputs include `(10, 7)` arrays
`hippo_L_subject_time.npy`, `hippo_R_subject_time.npy`, and
`hippo_bilateral_subject_time.npy`, plus CSV tables and PNG curves.

**Step 3: Predict hippocampal activity independently of Step 1**

```bash
python 03_predict_hippocampal_activity.py \
  --encoding-scores simulated_data/step3_prediction/encoding_scores.npy \
  --ifg-indices simulated_data/step3_prediction/ifg_indices.npy \
  --itg-indices simulated_data/step3_prediction/itg_indices.npy \
  --mfg-indices simulated_data/step3_prediction/mfg_indices.npy \
  --hippocampus simulated_data/step3_prediction/hippo_L_subject_time.npy \
  --timepoints simulated_data/step3_prediction/timepoints_order.npy \
  --output-dir results_smoke/prediction_manual \
  --n-permutations 20 \
  --seed 2026
```

The expected time-resolved output table has **seven rows**, and
`SVR_null_distribution_timepoint_by_perm.npy` has shape
`(7, n_permutations)`.

To test a realistic **Step 2 → Step 3 file hand-off**, replace the
`--hippocampus` and `--timepoints` inputs above with:

```text
results_smoke/hippocampus_manual/hippo_L_subject_time.npy
results_smoke/hippocampus_manual/timepoints_order.npy
```

The test runner checks this hand-off automatically.

### 7.4 What a passing smoke test does and does not establish

- **PASS means:** imports completed for the tested path, inputs had
  compatible dimensions, the script finished without an uncaught error,
  and expected outputs were produced with the right shapes.
- **PASS does not mean:** the simulated effects are biologically
  meaningful; the analysis reproduces manuscript figures; the original
  statistical assumptions are valid; or the true model reproduces results
  on real participant data.
- **Step 1 mock mode:** does not execute `tikreg`. It tests the surrounding
  pipeline and executes the original FDR calculation bodies extracted from
  the helper source while avoiding unused neuroimaging imports.
- **Small permutation count:** `10` or `20` permutations is for smoke
  testing only. Use the justified full analysis settings (e.g., 1,000 or
  more permutations, as specified by the manuscript) for real inference.
- **No participant data:** all arrays are intentionally simulated and
  should not be interpreted as observations from children.

Before using the scripts to support a code-availability or reproducibility
claim in a manuscript, also run **real Step 1** with the full dependency
environment and document the package versions, dataset provenance,
analysis parameters, and successful execution log.
