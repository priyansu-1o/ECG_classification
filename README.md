# ECG Classification: Synthetic Benchmark, Real MIT-BIH Validation and Inter-Patient Study

Course project for **Introduction to Data Science for Engineers (CS3006)**, IIITDM Kancheepuram, Team 21.

The project studies ECG heartbeat classification with classical machine learning, judged on **unseen patients** rather than on random beat splits. It has two linked parts and two supporting baselines:

| Part | Question | Data | Notebook |
|---|---|---|---|
| **A** | Which of six classifiers is best on an imbalanced synthetic ECG feature set, and does it transfer to real signals? | Synthetic features (1,000 samples), then real MIT-BIH DS2 segments | `CS24I1004_Varun.ipynb` |
| **B** | Can we replicate the inter-patient random-forest baseline of Saenz-Cogollo & Agelli (2020) on MIT-BIH, and do newer ideas improve it? | MIT-BIH Arrhythmia Database, AAMI classes N / S / V / F / Q | `CS24I1016_Priyansu.ipynb` |
| Baseline 1 | Patient-grouped LR and RF on simple RR and amplitude features | MIT-BIH | `CS24I1014_Ambadas.ipynb` |
| Baseline 2 | "Model A": LR and RF on 38 engineered features (**sample-stratified, not patient-independent**) | Released segment dataset without patient IDs | `CS24I1002_Hitesh.ipynb` |

> **Do not compare the notebooks' scores directly.** They use different datasets, features, class sets and splits (see [Protocols](#protocols-differ-across-notebooks)). Only Part B follows the published DS1/DS2 protocol with all five AAMI classes counted in evaluation.

---

## Headline results (Part B, MIT-BIH, DS2 = 22 unseen records)

Random Forest on the paper's top-6 features reproduces the reference result:

| DS2 metric | Ours | Reference (Saenz-Cogollo & Agelli 2020) |
|---|---|---|
| F1 class N | 0.977 | 0.980 |
| F1 class S | 0.690 | 0.731 |
| F1 class V | 0.905 | 0.909 |
| Accuracy | 0.956 | 0.961 |

All pre-declared models, each evaluated once on DS2 (F and Q beats counted as errors):

| Model (top-6 features) | DS1 LOPO macro-F1 | F1 N | F1 S | F1 V | Macro N/S/V | S recall rec. 232 | S recall others |
|---|---|---|---|---|---|---|---|
| RF (replication) | 0.732 | 0.977 | 0.690 | 0.905 | 0.857 | 0.858 | 0.600 |
| LGB + RR context | 0.736 | 0.978 | 0.631 | 0.926 | 0.845 | 0.667 | 0.546 |
| GA stack + context | 0.720 | 0.978 | 0.594 | 0.893 | 0.822 | 0.592 | 0.418 |
| XGB + RR context | 0.728 | 0.976 | 0.552 | 0.930 | 0.819 | 0.503 | 0.551 |
| RF + RR context | 0.742 | 0.977 | 0.568 | 0.900 | 0.815 | 0.548 | 0.469 |
| ET + RR context | 0.757 | 0.971 | 0.308 | 0.906 | 0.728 | 0.137 | 0.529 |

**Main findings**

1. **Problem formulation first.** Classifying 10-beat windows into brady / normal / tachycardia scored 97-100% only because the label is computed from the features; without them macro-F1 fell to 0.45 (LR) and 0.32 (RF).
2. **Patient-relative QRS widths were the decisive features.** They cut N-to-V false positives from 3,893 to 153 and lifted V precision from 0.41 to 0.93.
3. **The extensions did not give a reliable gain.** RR-context features helped slightly on DS1 leave-one-patient-out but hurt S on DS2 (0.690 to 0.568) and in 44-record LOPO. Patient-relative shape features, boosting, GA stacking and S-probability weighting did not beat the plain RF. Giving models more patient-specific features generalised worse.
4. **Class S is patient-dependent.** Records 232 and 222 hold 87% of DS2's S beats. A bootstrap over records gives an S-F1 95% interval of [0.11, 0.86] for the RF, so small differences between models are noise.

Part A has no recorded numeric results yet (its notebook has no executed outputs).

---

## Repository layout

Adjust to match your repository; this is the layout the notebooks assume.

```
.
├── README.md
├── CS24I1004_Varun.ipynb        # Part A: synthetic benchmark + real-data transfer pipeline
├── CS24I1016_Priyansu.ipynb         # Part B: full MIT-BIH experiment record
├── CS24I1014_Ambadas.ipynb      # Baseline 1: patient-grouped LR / RF on MIT-BIH
├── CS24I1002_Hitesh.ipynb       # Baseline 2: Model A explained notebook
```

---

## Setup

Python 3.10+ is recommended.

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib pyarrow \
            wfdb PyWavelets xgboost lightgbm shap ydata-profiling \
            imbalanced-learn jupyter
```

`imbalanced-learn` is needed for the SMOTE step of Part A. Baselines 1 and 2 also need the project files they import (`preprocessing.py`, `modeling.py`, `config/`, and `src/features`, `src/data/split`, `src/models/model_a`, `src/evaluation`), which are not part of the notebooks themselves.

### MIT-BIH data (Parts B and Baseline 1)

Download the raw files once:

```bash
pip install awscli
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ mitdb/
```

or, more slowly, in Python: `wfdb.dl_database('mitdb', 'mitdb/')`. You should end up with about 150 files (`100.dat`, `100.hea`, `100.atr`, ...).

---

## How to run

### Part B: `CS24I1016_Priyansu.ipynb`
Run the notebook top to bottom from the folder containing `mitdb/`. On Google Colab it mounts Drive and stores data and caches under `/content/drive/MyDrive/ecg_project`.

* Expensive steps are cached as parquet in `artifacts/` (`train_beats`, `test_beats`, `qrs_*`, `ctx_*`) and the out-of-fold probabilities in `oof.pkl`. Set `FORCE_REBUILD = True` to recompute everything.
* Rough run time on the first pass: feature extraction a few minutes (parallel); LOPO sweeps and out-of-fold probabilities 20-40 minutes; the GA exhaustive check 10-15 minutes.
* All random seeds are fixed (`random_state=0`).

Notebook sections: setup, raw-signal EDA, Experiment 1 (heart-rate windows, a negative result), feature engineering to parquet, feature-level EDA (quality audit, profiling, correlation, patient structure, DS1-vs-DS2 drift), evaluation protocol, Experiments 2-6, final DS2 comparison, robustness checks (44-record LOPO, bootstrap over records, per-record S recall), SHAP, conclusions.

### Part A: `CS24I1004_Varun.ipynb`
The notebook is a converted command-line script. Run it as a script, or replace `parser.parse_args()` with `parser.parse_args([])` when running inside Jupyter.

```bash
python <pipeline_script>.py                 # full pipeline incl. real MIT-BIH validation (default)
python <pipeline_script>.py --no-real       # synthetic benchmark and diagnostics only
```

Steps: generate synthetic data (1,000 samples, class ratios 0.70 / 0.15 / 0.15, `noise_level=1.0`, `seed=42`), patient-grouped DS1/DS2 split, six-model benchmark (GroupKFold with 5 splits, SMOTE, `RandomizedSearchCV` with `n_iter=8`), diagnostics on the best model (SHAP, ablation, noise sweep 0.0-3.0, probability calibration, error analysis), real MIT-BIH DS2 transfer (up to 25 segments per record), and export.

Outputs: `model_comparison.csv`, `shap_feature_importance.png`, `ablation_study.png`, `robustness_curve.png`, `probability_calibration.png`, `error_analysis.csv`, `real_mitdb_features.csv`, `domain_gap*` files, `best_rf_model.pkl` (note: saved under this name whichever model wins) and `ds2_predictions.csv`.

### Baselines
* **Ambadas:** `python preprocessing.py --data-dir mitdb --output data/processed/features.csv`, then `python modeling.py --features data/processed/features.csv --output-dir outputs/smoke --quick`. The notebook reads saved metrics unless `RUN_TRAINING = True`.
* **Hitesh:** run from the project root that contains `DataScience-Project/src`. Tuning can take several minutes.

---

## Protocols differ across notebooks

| | Part B (Priyansu) | Baseline 1 (Ambadas) | Baseline 2 (Hitesh) | Part A (Varun) |
|---|---|---|---|---|
| Data | MIT-BIH, 44 records | MIT-BIH | Released segments, no patient IDs | Synthetic, then real DS2 segments |
| Split | Fixed DS1/DS2; LOPO for selection | DS1/DS2 with 201 and 202 both in DS1; grouped CV | Sample-stratified, 100-sample test | Patient-grouped DS1/DS2; GroupKFold |
| Classes trained | N, S, V (F and Q counted as errors at test) | all five | all five | three synthetic classes |
| Features | 124 (Stage-1, QRS, context) | RR and amplitude | 38 engineered | synthetic features |

Quick-run scores reported in the baselines are not comparable with Part B: Baseline 1 reports DS2 accuracy 93.8% (LR) / 85.9% (RF) and macro-F1 0.395 / 0.386 over five classes; Baseline 2 reports 92% / 96% accuracy on a small sample-level test set, which is not patient-independent.

---

## Team 21

| Name | Roll no. | Notebook |
|---|---|---|
| Varun Sharma Garimella | CS24I1004 | Part A pipeline |
| Hitesh Kalbande | CS24I1002 | Baseline 2 (Model A) |
| Priyansu Samanta | CS24I1016 | Part B |
| Ambadas Rautrao | CS24I1014 | Baseline 1 |

## References

* J. F. Saenz-Cogollo and M. Agelli, "Investigating feature selection and random forests for inter-patient heartbeat classification," *Algorithms* 13(4):75, 2020.
* P. de Chazal, M. O'Dwyer, R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Trans. Biomed. Eng.* 51(7):1196-1206, 2004.
* G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Eng. Med. Biol. Mag.* 20(3):45-50, 2001.
* C. Zou et al., "Heartbeat classification by random forest with a novel context feature: a segment label," *IEEE J. Transl. Eng. Health Med.* 10:1900508, 2022.
* N. Zhao et al., "Improved stacked ensemble with genetic algorithm for automatic ECG diagnosis of children living in high-altitude areas," *Biomed. Signal Process. Control* 87:105506, 2024.
* A. J. Prakash et al., "Advances in machine and deep learning for ECG beat classification: a systematic review," *Front. Digit. Health* 7:1649923, 2025.

The full bibliography is in the project report.
