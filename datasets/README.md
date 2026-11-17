# Datasets

The datasets are not redistributed. Download each one, convert it to the
format below, and save it under this folder with the file name given in
`datasets_config.yaml`.

What the scripts do with each file (`load_dataset()` in
`run_calibration_nn.py`): read the CSV, lower-case the column names and
replace spaces with underscores, oversample the minority class to balance it
(seed 42), split 70/30 (seed 42), then bin every remaining column into three
quantile bins. **Every column other than the target must therefore be
numeric**, and the target must be binary.

| file | dataset | source | rows | target column | target coding |
|---|---|---|---:|---|---|
| `cancer.csv` | Breast Cancer Wisconsin (Diagnostic) | [UCI #17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | `diagnosis` | 0 = malignant, 1 = benign |
| `compas.csv` | COMPAS two-year recidivism, ProPublica data as prepared for FairML | [Kaggle `danofer/compass`](https://www.kaggle.com/datasets/danofer/compass) (`propublica_data_for_fairml.csv`), from [ProPublica](https://github.com/propublica/compas-analysis) | 6,172 | `two_yr_recidivism` | 0 / 1 |
| `diabetes.csv` | Pima Indians Diabetes | [Kaggle `uciml/pima-indians-diabetes-database`](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | `outcome` | 0 / 1 |
| `employee.csv` | Employee attrition | [Kaggle `tawfikelmetwally/employee-dataset`](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset) | 4,653 | `leave_or_not` | 0 = stays, 1 = leaves |
| `heart_failure_clinical.csv` | Heart failure clinical records | [UCI #519](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) | 299 | `death_event` | 0 = survived, 1 = died |

Notes per dataset:

- **cancer**: drop the `id` column (and the empty trailing column of some
  copies); recode `diagnosis` M → 0, B → 1. The 30 remaining columns are the
  numeric features.
- **compas**: the FairML file is already numeric (12 columns). Used only as a
  degenerate control in the paper.
- **diabetes**, **heart**: used as distributed; only the column names are
  normalised by the loader.
- **employee**: rename the target to `leave_or_not` and encode the categorical
  columns (`Education`, `City`, `Gender`, `EverBenched`) as integers.

The datasets remain under their own licences and terms of use.
