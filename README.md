# Classical ML Auto-Framework (V4)

A beginner-friendly Python project that helps you run **classical machine-learning experiments** on your own tabular datasets.

You do not need to write a separate training script for every algorithm. You choose the type of ML task, tell the framework where your dataset is, identify the target column (when needed), and run the Python file. The framework handles much of the repetitive work: preprocessing, model comparison, optional tuning, evaluation, and saving results.

> **Important:** This is an educational automation framework, not a magic button. It cannot guarantee a good model. You still need to understand your dataset, check for data leakage, read the metrics, and decide whether the result makes sense.

---

## 1. What does this project do?

V4 supports three kinds of machine-learning tasks:

| Task | Meaning | Example |
|---|---|---|
| `classification` | Predict a category or label | Predict whether an email is spam or not |
| `regression` | Predict a numerical value | Predict a house's sale price |
| `unsupervised` | Find patterns when you do not provide a target column | Group similar customer records |

### Main features

- Reads data from **CSV, XLSX, and XLS** files.
- Automatically separates numeric, categorical, and selected text columns.
- Handles missing values using preprocessing pipelines.
- Scales numeric features and one-hot encodes categorical features where applicable.
- Compares multiple classification or regression models using cross-validation.
- Can optionally tune selected models with randomized hyperparameter search.
- Selects supervised models using training cross-validation scores, then evaluates the selected model on the holdout test set.
- Saves the best model and experiment reports.
- Includes optional feature selection.
- Includes TF-IDF support for text columns in supervised tasks.
- Can optionally add sentiment columns using FinBERT.
- Offers optional clustering, dimensionality-reduction, and anomaly-detection analyses.
- Has a prediction mode for applying a previously saved model to a new dataset.
- Creates a dataset audit with row counts, duplicate counts, data types, missing values, and unique-value counts.

---

## 2. Project files

Your GitHub repository can have this simple structure:

```text
Classical-ML-Auto-Framework/
│
├── classical_ml_v4.py
├── README.md
├── requirements.txt
├── requirements-finbert.txt
├── .gitignore
│
└── (Your dataset goes here locally, but do not upload private data)
```

Rename your downloaded script to **`classical_ml_v4.py`** before adding it to the repository.

The folders and files inside `ml_results_v4/` are generated when you run the framework. They do not need to be committed to GitHub.

---

## 3. Requirements

- Python 3.10 or newer is recommended.
- pip (Python's package installer).
- A CSV or Excel dataset.
- An internet connection is needed the first time you install packages. FinBERT also downloads its model files the first time it is used.

You do **not** need a GPU for the normal classical ML features. FinBERT can use a supported CUDA GPU when available, but it can also run on CPU.

---

## 4. Installation

Open a terminal in the project folder.

### Step 1 — (Recommended) Create a virtual environment

A virtual environment keeps this project's Python packages separate from other projects.

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Step 2 — Install the required packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional — Install FinBERT dependencies

Only do this if you want to enable the optional FinBERT sentiment feature:

```bash
pip install -r requirements-finbert.txt
```

---

## 5. Prepare your dataset

For a first run, use a small, clean educational dataset. For example, imagine a CSV file named `students.csv`:

| study_hours | attendance | previous_score | result |
|---:|---:|---:|---|
| 2 | 70 | 55 | Fail |
| 5 | 85 | 72 | Pass |
| 3 | 75 | 60 | Pass |
| 1 | 50 | 40 | Fail |

Here:
- Each **row** represents one observation (one student in this example).
- Each **column** represents a feature or a target.
- `study_hours`, `attendance`, and `previous_score` are features.
- `result` is the target we want the model to predict.

Your actual dataset must contain enough useful rows and more than one target value for supervised training. Very small datasets may not support reliable model comparison or cross-validation.

---

## 6. Configure the framework

Open `classical_ml_v4.py` and find the section:

```python
# 1. CONFIG - EDIT THESE SETTINGS
```

For a normal supervised task, the most important settings are:

```python
DATA_PATH = "students.csv"
TASK = "classification"
TARGET_COLUMN = "result"
```

Change these values to match your own dataset:

- `DATA_PATH`: path to your dataset. If the CSV is in the same folder as the Python file, its filename is enough, such as `"students.csv"`.
- `TASK`: choose `"classification"`, `"regression"`, or `"unsupervised"`.
- `TARGET_COLUMN`: write the exact target column name, including spelling and capital letters. This setting is ignored for unsupervised analysis.

### A few more useful settings

```python
OUTPUT_DIR = "ml_results_v4"
TEST_SIZE = 0.20
RANDOM_STATE = 42

RUN_CROSS_VALIDATION = True
RUN_HYPERPARAMETER_TUNING = True
TUNE_TOP_N_MODELS = 3
TUNING_ITERATIONS = 12
CV_FOLDS = 5

ENABLE_FEATURE_SELECTION = False
```

- `OUTPUT_DIR`: folder where reports, models, and predictions are saved.
- `TEST_SIZE = 0.20`: reserves 20% of the usable rows as a final test set.
- `RANDOM_STATE`: helps make random splits repeatable.
- `RUN_CROSS_VALIDATION`: keep this enabled for supervised model selection. V4 requires usable cross-validation scores to select a supervised model.
- `RUN_HYPERPARAMETER_TUNING`: enables an additional search for better model settings. It can increase runtime.
- `TUNE_TOP_N_MODELS`: number of top baseline models considered for tuning.
- `TUNING_ITERATIONS`: number of parameter combinations sampled for each tuning search.
- `CV_FOLDS`: requested number of cross-validation folds. The framework may reduce this when the dataset has fewer examples in its smallest class.
- `ENABLE_FEATURE_SELECTION`: optionally selects a subset of features.

For your first experiment, it is sensible to keep most defaults unchanged and only edit the dataset path, task, and target column.

---

## 7. Run the project

After saving your configuration, run:

```bash
python classical_ml_v4.py
```

The framework will:
1. Load your dataset.
2. Check the dataset and create an audit report.
3. Prepare the features.
4. Compare suitable models using cross-validation.
5. Optionally tune selected models.
6. Select a model using cross-validation results.
7. Evaluate the selected model on the holdout test set.
8. Save the model and reports.

The first run can take some time, especially when model tuning is enabled or the dataset is large.

---

## 8. How to use the three task types

### A. Classification

Use this when the target is a category, such as `Yes/No`, `Pass/Fail`, or a class name.

```python
TASK = "classification"
TARGET_COLUMN = "result"
```

### B. Regression

Use this when the target is a continuous numerical value, such as a score, temperature, or price in a non-financial educational dataset.

```python
TASK = "regression"
TARGET_COLUMN = "temperature"
```

### C. Unsupervised learning

Use this when you do not have a target column and want to explore patterns in the data.

```python
TASK = "unsupervised"
TARGET_COLUMN = ""
```

For unsupervised mode, the framework can run clustering, dimensionality-reduction, and anomaly-detection analyses according to the configuration switches in the script.

---

## 9. Working with text data (NLP)

V4 supports TF-IDF text features for supervised classification and regression.

For example, if your CSV has a column named `review_text`, add that exact column name:

```python
TEXT_COLUMNS = ["review_text"]
```

For multiple text columns:

```python
TEXT_COLUMNS = ["review_text", "customer_comment"]
```

The target column must not also be listed in `TEXT_COLUMNS`. The current V4 implementation does not support TF-IDF text columns in unsupervised mode.

---

## 10. Optional FinBERT sentiment feature

FinBERT is an optional language model that can add positive, negative, and neutral sentiment scores for a text column.

Install the optional dependencies first, then configure:

```python
ENABLE_FINBERT_SENTIMENT = True
FINBERT_TEXT_COLUMN = "text"
```

Replace `"text"` with the exact name of your text column.

The first run may download the FinBERT model, so it can take longer and requires internet access. FinBERT sentiment scores describe text sentiment; **they are not price forecasts, trading signals, or investment advice**.

---

## 11. Predict using a saved model

First, train a model normally. The framework saves the selected pipeline at:

```text
ml_results_v4/best_model.joblib
```

To predict on a new dataset, edit the configuration:

```python
PREDICT_MODE = True
SAVED_MODEL_PATH = "ml_results_v4/best_model.joblib"
NEW_DATA_PATH = "new_data.csv"
PREDICTION_OUTPUT = "ml_results_v4/predictions.csv"
```

Make sure the new file contains the same feature columns used during training. For supervised prediction, do not include the target column. V4 checks the saved feature schema and reports missing required columns.

Then run the script again:

```bash
python classical_ml_v4.py
```

Predictions will be written to the configured output CSV file.

After prediction mode is enabled, switch `PREDICT_MODE` back to `False` when you want to train another model.

---

## 12. Where are the results saved?

By default, the framework creates an `ml_results_v4/` folder. Depending on the task and which optional features are enabled, it can contain:

- `best_model.joblib` — the selected trained pipeline.
- `run_metadata.json` — task, feature information, model details, and evaluation metrics.
- `feature_schema.json` — feature-column information used for prediction checks.
- `data_audit.json` and `data_audit_columns.csv` — dataset audit information.
- Model-comparison CSV files.
- Classification report and confusion-matrix files for classification.
- Tuning and individual model files.
- Prediction CSV files.
- Clustering, dimensionality-reduction, or anomaly-detection outputs in unsupervised mode.

These files are generated outputs. Keep them locally unless you have a specific reason to publish a small, non-sensitive example.

---

## 13. Important limitations

- Automated model comparison does not replace understanding the data or the problem.
- A high score does not automatically mean a model will perform well on new, real-world data.
- Cross-validation and holdout evaluation need enough representative data. Tiny or imbalanced datasets can produce unstable results.
- Some algorithms may fail on particular datasets or take a long time. The framework reports failures and continues where possible.
- Hyperparameter tuning can be computationally expensive. Reduce tuning iterations or disable tuning if runtime is too high.
- Text processing can create many features and use significant memory.
- FinBERT is optional and has additional package/model-download requirements.
- The current educational build has a guard that blocks datasets it identifies as financial-market data. It is not intended for market forecasting or trading.
- The script was syntax-checked, but that does not prove that every feature will work on every dataset. Test it with a suitable sample dataset before relying on its outputs.

---

## 14. Troubleshooting

**`FileNotFoundError`**
- Check `DATA_PATH` or `NEW_DATA_PATH`.
- If the data file is in the same folder as the script, use only its filename.
- Check that the file extension is supported.

**`TARGET_COLUMN not found`**
- Copy the exact column name from your CSV/Excel header.
- Check spaces, spelling, and capital letters.

**`No model produced a valid CV score` or cross-validation error**
- Check whether the dataset has enough rows.
- For classification, check how many examples exist in the smallest class.
- Do not disable cross-validation to bypass model selection.

**FinBERT import error**
- Install `requirements-finbert.txt` and try again.

**The run is too slow**
- Set `RUN_HYPERPARAMETER_TUNING = False`.
- Reduce `TUNE_TOP_N_MODELS` or `TUNING_ITERATIONS`.
- Start with a smaller dataset.

---

## 15. License

No license is included by default. If you want others to use, modify, and redistribute this project, choose a license and add its full license text in a file named `LICENSE`. Do not claim a license you have not actually added.

---

## Author

Add your name, GitHub profile, and any acknowledgements here before publishing.
