# Cortical Encoding of Knowledge Events During Naturalistic Learning

Analysis code accompanying the study *Cortical Encoding of Knowledge Events and Hippocampal Dynamics during Naturalistic Learning in Children*.

## Code overview

| Script | Purpose |
| --- | --- |
| `01_cortical_encoding_model.py` | Fit cortical encoding models using visual, auditory, character, semantic, and knowledge-event features. |
| `02_extract_hippocampal_activity.py` | Extract hippocampal activity around knowledge events. |
| `03_predict_hippocampal_activity.py` | Predict hippocampal activity from IFG, ITG, and MFG encoding scores using leave-one-subject-out SVR. |

Supporting modules: `encoding_helpers.py`, `fdr_correction_helpers.py`, and `data_prep_helpers_new.py`.

## Requirements

Python 3.10+ is recommended. Install the dependencies for the basic tests:

```bash
pip install -r requirements_smoke.txt
```

Running the **full cortical encoding model** also requires `tikreg`, `nibabel`, `nilearn`, and `brainiak`. Install versions compatible with your Python environment.

## Test with synthetic data

The `simulated_data/` folder contains small, **artificial** datasets for testing all three scripts. Regenerate them if needed:

```bash
python generate_simulated_data.py
```

Run the automated smoke test:

```bash
python smoke_test_pipeline.py --step1 mock --n-permutations 10
```

This checks the actual Steps 2 and 3, their file hand-off, and the Step 1 input/output and FDR interface. **For Step 1 only, `mock` replaces `tikreg` fitting with least squares**; it does not test the actual encoding estimator.

Once the full dependencies are installed, test the real Step 1 model:

```bash
python smoke_test_pipeline.py --step1 real --n-permutations 20
```

Results and the pass/fail report are saved in `results_smoke/SMOKE_TEST_REPORT.json`.

### Example input shapes

| Analysis | Example inputs |
| --- | --- |
| Step 1 | Cortical BOLD `(10, 18, 160)` and five feature arrays with 160 TRs each |
| Step 2 | Hippocampal time courses `(10, 10, 160)` and event labels `(160,)` |
| Step 3 | Cortical encoding scores `(10, 18)`, three ROI index files, and hippocampal targets `(10, 7)` |

Each script supports `--help` for its available command-line arguments. The `simulated_data/synthetic_manifest.json` file lists all test files and their shapes.

## Notes

- All test datasets are **synthetic**, not participant data. Successful execution does **not** establish scientific validity or reproduce the study's results.
- Use the original research data, correct ROI definitions, and full analysis settings for scientific analyses.
- The supplied Step 1 script includes a small correction to align permutation-null and parcel-wise p-value dimensions; the original FDR helper functions are unchanged.
