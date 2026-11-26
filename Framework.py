"""
CLASSICAL ML AUTO-FRAMEWORK V4
==============================

Run:
    pip install -U pandas numpy scikit-learn joblib openpyxl
    python classical_ml_v4.py

Edit only the CONFIG section.

What it does:
- Explicit task selection: classification / regression / unsupervised
- CSV and Excel loading
- Automatic numeric/categorical preprocessing inside leakage-safe pipelines
- Missing-value handling, scaling and one-hot encoding
- Cross-validation model comparison
- Optional feature selection and hyperparameter tuning
- Holdout test evaluation after model selection
- Class-imbalance-aware metrics and balanced class weights where supported
- Saves every successful model, comparison tables, reports and the best pipeline
- Optional prediction mode for a new CSV/Excel file
- Optional clustering, dimensionality reduction and anomaly detection
- Dataset audit report and stronger prediction-time feature alignment
- Corrected model selection: tuning candidates are ranked by CV, not holdout results
- TF-IDF NLP support for supervised text classification/regression
- Optional local FinBERT sentiment enrichment for text columns
- Holdout set is used only once, after CV-based model selection
- Stronger feature/schema validation for prediction mode
- Explicit guard against financial-market price/market datasets in this educational build

Important:
- Set TARGET_COLUMN for classification/regression.
- For unsupervised, TARGET_COLUMN is ignored and all columns are used as features.
- The framework is general-purpose, not magic: no script can guarantee good results
  on every dataset. Review the generated warnings, metrics and data leakage risks.
"""

from pathlib import Path
from datetime import datetime
import json
import warnings
import traceback

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_validate,
    RandomizedSearchCV
)
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, mutual_info_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, median_absolute_error,
    roc_auc_score, log_loss, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score
)

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier, IsolationForest,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, Birch, OPTICS, SpectralClustering,
    MeanShift
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, KernelPCA

import os

# =====================================================================
# 1. CONFIG - EDIT THESE SETTINGS
# =====================================================================

DATA_PATH = r"C:\Users\LENOVO\Desktop\train.csv"
TASK = "classification"       # "classification", "regression", or "unsupervised"
TARGET_COLUMN = ""      # exact column name; ignored for unsupervised

OUTPUT_DIR = "ml_results_v4"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Automation switches
RUN_CROSS_VALIDATION = True
RUN_HYPERPARAMETER_TUNING = True
TUNE_TOP_N_MODELS = 3
TUNING_ITERATIONS = 12
CV_FOLDS = 5
ENABLE_FEATURE_SELECTION = False
FEATURE_SELECTION_PERCENTILE = 80

# NLP: add the exact text columns from your CSV/Excel file here.
# Example: TEXT_COLUMNS = ["review", "comment", "news_text"]
TEXT_COLUMNS = []
TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)

# Optional FinBERT sentiment enrichment (local inference; first run downloads model).
# This analyzes text sentiment only; it does NOT forecast prices or generate trades.
ENABLE_FINBERT_SENTIMENT = False
FINBERT_TEXT_COLUMN = "text"
FINBERT_BATCH_SIZE = 16
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Runtime / safety controls
N_JOBS = -1
MAX_TUNING_ROWS = 50000
MAX_UNSUPERVISED_ROWS = 5000
MAX_DENSE_FEATURES = 15000
RUN_POLYNOMIAL_REGRESSION = True

# Optional unsupervised analyses
RUN_UNSUPERVISED_EXTRAS = False
RUN_CLUSTERING = True
RUN_DIMENSIONALITY_REDUCTION = True
RUN_ANOMALY_DETECTION = True

# Prediction mode: train a model first, then set this True in a later run.
PREDICT_MODE = False
SAVED_MODEL_PATH = "ml_results_v4/best_model.joblib"
NEW_DATA_PATH = "new_data.csv"
PREDICTION_OUTPUT = "ml_results_v4/predictions.csv"

# This build is intended for educational/non-financial datasets only.
# It is not a market forecasting or trading tool.
BLOCK_FINANCIAL_MARKET_DATA = True
FINANCIAL_MARKET_MARKERS = {
    "forex", "xauusd", "xau", "gold_price", "crypto_price", "stock_price",
    "ticker", "currency_pair", "bid", "ask", "spread", "ohlc"
}
OHLC_COLUMN_GROUP = {"open", "high", "low", "close"}


# =====================================================================
# 2. UTILITIES
# =====================================================================

def banner(message):
    print("\n" + "=" * 78)
    print(message)
    print("=" * 78)


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def load_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}\nChange DATA_PATH / NEW_DATA_PATH in CONFIG."
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError("Supported file types are CSV, XLSX and XLS.")


def validate_educational_dataset(df, source_path=None):
    """Reject clearly identifiable financial-market datasets in this build."""
    if not BLOCK_FINANCIAL_MARKET_DATA:
        return
    names = {str(c).strip().lower().replace(" ", "_") for c in df.columns}
    source = str(source_path or "").lower().replace("-", "_").replace(" ", "_")
    markers = FINANCIAL_MARKET_MARKERS.intersection(names)
    has_ohlc = OHLC_COLUMN_GROUP.issubset({str(c).strip().lower() for c in df.columns})
    if markers or has_ohlc or any(term in source for term in ("forex", "xau", "gold", "stock_price", "crypto_price")):
        raise ValueError(
            "This educational V3 build is limited to non-financial datasets. "
            "Use a harmless dataset such as Titanic, student performance, weather, "
            "or sensor readings instead."
        )

def save_data_audit(df, output_dir):
    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_detail": []
    }
    for column in df.columns:
        series = df[column]
        audit["columns_detail"].append({
            "column": str(column),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_fraction": float(series.isna().mean()),
            "unique_values": int(series.nunique(dropna=True))
        })
    save_json(audit, output_dir / "data_audit.json")
    pd.DataFrame(audit["columns_detail"]).to_csv(output_dir / "data_audit_columns.csv", index=False)
    print(f"Data audit saved (duplicate rows: {audit['duplicate_rows']}).")


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # compatibility with older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor(X, dense=False, scale_numeric=True):
    text_columns = [c for c in TEXT_COLUMNS if c in X.columns]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric and c not in text_columns]

    transformers = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))

    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe())
        ])
        transformers.append(("categorical", categorical_pipe, categorical))

    for index, column in enumerate(text_columns):
        text_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32
        )
        # ColumnTransformer passes a 1-D array for a scalar column selector.
        transformers.append((f"text_{index}", text_vectorizer, column))

    if not transformers:
        raise ValueError("No usable feature columns remain.")
    if dense and text_columns:
        raise ValueError("Dense-only estimators are disabled when TF-IDF text columns are configured.")

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense else 1.0
    )


def make_pipeline(X, estimator, task, dense=False, scale_numeric=True):
    steps = [("preprocess", make_preprocessor(X, dense=dense, scale_numeric=scale_numeric))]
    if ENABLE_FEATURE_SELECTION:
        score_func = mutual_info_classif if task == "classification" else mutual_info_regression
        steps.append((
            "feature_selection",
            SelectPercentile(score_func=score_func, percentile=FEATURE_SELECTION_PERCENTILE)
        ))
    steps.append(("model", estimator))
    return Pipeline(steps)


def safe_cv_splits(y, task):
    if task == "classification":
        counts = pd.Series(y).value_counts()
        smallest_class = int(counts.min())
        splits = min(CV_FOLDS, smallest_class)
        if splits < 2:
            return None, "At least one class has fewer than 2 samples; CV is disabled."
        return StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE), None
    splits = min(CV_FOLDS, len(y))
    if splits < 2:
        return None, "Not enough rows for cross-validation."
    return KFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE), None


def save_table(rows, path):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


# =====================================================================
# 3. MODEL CATALOGUE
# =====================================================================

def get_regression_catalogue():
    sparse = {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "Lasso": (Lasso(max_iter=15000), {"model__alpha": [0.001, 0.01, 0.1, 1.0]}),
        "ElasticNet": (ElasticNet(max_iter=15000), {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }),
        "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), {
            "model__max_depth": [None, 4, 8, 15, 25],
            "model__min_samples_split": [2, 5, 10]
        }),
        "Random Forest": (RandomForestRegressor(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ), {
            "model__max_depth": [None, 8, 16, 25],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt", 0.7]
        }),
        "Extra Trees": (ExtraTreesRegressor(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ), {
            "model__max_depth": [None, 8, 16, 25],
            "model__min_samples_leaf": [1, 2, 4]
        }),
        "KNN Regressor": (KNeighborsRegressor(), {
            "model__n_neighbors": [3, 5, 7, 9, 13],
            "model__weights": ["uniform", "distance"]
        }),
        "SVR": (SVR(), {
            "model__C": [0.1, 1, 10, 50],
            "model__epsilon": [0.01, 0.1, 0.2],
            "model__kernel": ["rbf", "linear"]
        }),
    }
    dense = {
        "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_STATE), {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4]
        }),
        "Hist Gradient Boosting": (HistGradientBoostingRegressor(random_state=RANDOM_STATE), {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [100, 200],
            "model__max_leaf_nodes": [15, 31]
        }),
    }
    return sparse, dense


def get_classification_catalogue():
    sparse = {
        "Logistic Regression": (LogisticRegression(max_iter=4000), {
            "model__C": [0.01, 0.1, 1, 10, 50],
            "model__class_weight": [None, "balanced"]
        }),
        "KNN Classifier": (KNeighborsClassifier(), {
            "model__n_neighbors": [3, 5, 7, 9, 13],
            "model__weights": ["uniform", "distance"]
        }),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), {
            "model__max_depth": [None, 4, 8, 15, 25],
            "model__class_weight": [None, "balanced"]
        }),
        "Random Forest": (RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ), {
            "model__max_depth": [None, 8, 16, 25],
            "model__min_samples_leaf": [1, 2, 4],
            "model__class_weight": [None, "balanced", "balanced_subsample"]
        }),
        "Extra Trees": (ExtraTreesClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ), {
            "model__max_depth": [None, 8, 16, 25],
            "model__class_weight": [None, "balanced"]
        }),
        "SVC": (SVC(), {
            "model__C": [0.1, 1, 10, 50],
            "model__kernel": ["rbf", "linear"],
            "model__class_weight": [None, "balanced"]
        }),
        "SGD Classifier": (SGDClassifier(random_state=RANDOM_STATE), {
            "model__loss": ["hinge", "log_loss"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__class_weight": [None, "balanced"]
        }),
        "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE), {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.03, 0.1, 0.5, 1.0]
        }),
    }
    dense = {
        "Gaussian Naive Bayes": (GaussianNB(), {
            "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4]
        }),
        "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=RANDOM_STATE), {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [100, 200],
            "model__max_leaf_nodes": [15, 31]
        }),
    }
    return sparse, dense


# =====================================================================
# 4. CROSS-VALIDATION, TUNING AND HOLDOUT EVALUATION
# =====================================================================

def regression_metrics(y_true, pred):
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "Median_AE": float(median_absolute_error(y_true, pred)),
        "R2": float(r2_score(y_true, pred)) if len(y_true) > 1 else np.nan
    }


def classification_metrics(y_true, pred, model, X_test):
    values = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_weighted": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            classes = model.named_steps["model"].classes_
            if len(classes) == 2:
                values["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            else:
                values["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
                )
            values["log_loss"] = float(log_loss(y_true, probabilities, labels=classes))
    except Exception:
        pass
    return values


def run_supervised(df, output_dir, task):
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"TARGET_COLUMN '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    data = df.dropna(subset=[TARGET_COLUMN]).copy()
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    for text_column in TEXT_COLUMNS:
        if text_column in X.columns:
            X[text_column] = X[text_column].fillna("").astype(str)
    if TARGET_COLUMN in TEXT_COLUMNS:
        raise ValueError("TARGET_COLUMN cannot also be listed in TEXT_COLUMNS.")
    if not 0.05 <= TEST_SIZE <= 0.5:
        raise ValueError("TEST_SIZE must be between 0.05 and 0.50.")

    if X.empty:
        raise ValueError("No feature columns remain after removing the target.")
    if y.nunique() < 2:
        raise ValueError("Target must contain at least 2 distinct values.")
    if len(data) < 10:
        raise ValueError("Dataset has fewer than 10 usable rows; model comparison is unreliable.")

    # Drop constant feature columns.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)

    if task == "classification":
        counts = y.value_counts()
        if counts.min() < 2:
            raise ValueError(
                "Every class needs at least 2 examples for a stratified train/test split. "
                "Add data or merge/remove extremely rare labels."
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    print(f"Rows: {len(data)} | Features: {X.shape[1]}")
    print(f"Train: {len(X_train)} | Holdout test: {len(X_test)}")
    print(f"Task: {task.upper()}")

    if task == "classification":
        sparse_catalogue, dense_catalogue = get_classification_catalogue()
        scoring = "f1_weighted"
        sort_ascending = False
    else:
        sparse_catalogue, dense_catalogue = get_regression_catalogue()
        scoring = "neg_root_mean_squared_error"
        sort_ascending = False

    catalogue = []
    for name, (estimator, params) in sparse_catalogue.items():
        catalogue.append((name, estimator, params, False))
    if not any(c in X_train.columns for c in TEXT_COLUMNS):
        for name, (estimator, params) in dense_catalogue.items():
            catalogue.append((name, estimator, params, True))
    else:
        print("NLP mode: TF-IDF enabled; dense-only models are skipped to avoid memory blow-ups.")

    if not RUN_CROSS_VALIDATION:
        raise ValueError("V4 requires RUN_CROSS_VALIDATION=True so model selection never uses the holdout set.")
    cv, cv_warning = safe_cv_splits(y_train, task)
    if cv_warning or cv is None:
        raise ValueError(
            "Reliable training-only cross-validation is not possible: "
            + str(cv_warning or "CV unavailable")
            + " Add more examples per class/row before training."
        )

    rows = []
    fitted = {}
    cv_ranking = []

    banner("STAGE 1 - BASELINE MODEL COMPARISON")
    for name, estimator, params, dense in catalogue:
        try:
            pipe = make_pipeline(X_train, estimator, task, dense=dense)
            cv_score = np.nan
            cv_std = np.nan

            if RUN_CROSS_VALIDATION and cv is not None:
                scores = cross_validate(
                    pipe, X_train, y_train, cv=cv, scoring=scoring,
                    n_jobs=1, error_score="raise", return_train_score=False
                )
                cv_score = float(np.mean(scores["test_score"]))
                cv_std = float(np.std(scores["test_score"]))

            # Fit only on training data. The holdout partition is not inspected here.
            pipe.fit(X_train, y_train)
            row = {
                "model": name,
                "cv_score": cv_score,
                "cv_std": cv_std,
                "status": "OK"
            }
            rows.append(row)
            fitted[name] = pipe
            cv_ranking.append((name, cv_score, params, dense))
            print(f"{name:30s} CV={cv_score:.5f} (+/- {cv_std:.5f})")
        except Exception as exc:
            print(f"{name:30s} FAILED: {exc}")
            rows.append({"model": name, "status": f"FAILED: {exc}"})

    if not fitted:
        raise RuntimeError("All candidate models failed. Check dataset types and error messages.")

    # Candidate selection is based exclusively on training-partition CV scores.
    successful = [r for r in rows if r.get("status") == "OK" and pd.notna(r.get("cv_score"))]
    if not successful:
        raise RuntimeError("No model produced a valid CV score; refusing to select on holdout data.")
    ranking = sorted(successful, key=lambda r: r["cv_score"], reverse=True)

    # Stage 2: tune only the best few baseline candidates to keep runtime bounded.
    tuned_candidates = []
    if RUN_HYPERPARAMETER_TUNING and len(X_train) <= MAX_TUNING_ROWS and cv is not None:
        banner("STAGE 2 - HYPERPARAMETER TUNING")
        names_to_tune = [row["model"] for row in ranking[:TUNE_TOP_N_MODELS]]
        for name in names_to_tune:
            match = next(item for item in catalogue if item[0] == name)
            _, estimator, params, dense = match
            if not params:
                print(f"{name}: no tuning grid; keeping baseline.")
                continue
            try:
                base_pipe = make_pipeline(X_train, estimator, task, dense=dense)
                search = RandomizedSearchCV(
                    base_pipe,
                    param_distributions=params,
                    n_iter=TUNING_ITERATIONS,
                    scoring=scoring,
                    cv=cv,
                    random_state=RANDOM_STATE,
                    # Avoid nested parallelism with estimators that already use N_JOBS.
                    n_jobs=1,
                    refit=True,
                    error_score=np.nan
                )
                search.fit(X_train, y_train)
                tuned = search.best_estimator_
                tuned_name = name + " [TUNED]"
                tuned_cv_score = float(search.best_score_)
                if np.isfinite(tuned_cv_score):
                    tuned_candidates.append((tuned_cv_score, tuned_name, tuned))
                joblib.dump(tuned, output_dir / f"tuned_{safe_name(name)}.joblib")
                print(f"{name}: best CV score={tuned_cv_score:.5f}")
                print("  Best parameters:", search.best_params_)
            except Exception as exc:
                print(f"TUNING FAILED for {name}: {exc}")
    elif RUN_HYPERPARAMETER_TUNING:
        print(
            f"Tuning skipped: training rows exceed MAX_TUNING_ROWS "
            f"({MAX_TUNING_ROWS}) or CV unavailable."
        )

    # Save every successful baseline model.
    for name, model in fitted.items():
        joblib.dump(model, output_dir / f"baseline_{safe_name(name)}.joblib")

    # Select baseline and tuned candidates using training-only CV scores.
    baseline_winner_name = ranking[0]["model"]
    best_model_name = baseline_winner_name
    best_model = fitted[baseline_winner_name]
    best_selection_score = float(ranking[0]["cv_score"])

    if tuned_candidates:
        tuned_candidates.sort(key=lambda item: item[0], reverse=True)
        tuned_cv_score, tuned_name, tuned_model = tuned_candidates[0]
        if tuned_cv_score > best_selection_score:
            best_model_name = tuned_name
            best_model = tuned_model
            best_selection_score = tuned_cv_score

    # The holdout set is evaluated exactly once, after model selection.
    final_prediction = best_model.predict(X_test)
    if task == "regression":
        final_metrics = regression_metrics(y_test, final_prediction)
    else:
        final_metrics = classification_metrics(y_test, final_prediction, best_model, X_test)
    best_holdout_metrics = {
        "model": best_model_name,
        "cv_score": best_selection_score,
        **final_metrics
    }

    joblib.dump(best_model, output_dir / "best_model.joblib")
    for row in rows:
        if row.get("status") == "OK":
            row["selected_by_cv"] = row["model"] == baseline_winner_name
    save_table(rows, output_dir / f"{task}_all_model_results.csv")

    # Detailed final report for the selected model.
    if task == "classification":
        report = classification_report(y_test, final_prediction, output_dict=True, zero_division=0)
        save_json(report, output_dir / "best_model_classification_report.json")
        labels = list(pd.unique(pd.concat([pd.Series(y_test), pd.Series(final_prediction)])))
        cm = confusion_matrix(y_test, final_prediction, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            output_dir / "best_model_confusion_matrix.csv"
        )

    metadata = {
        "task": task,
        "target_column": TARGET_COLUMN,
        "data_path": DATA_PATH,
        "best_model": best_model_name,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "feature_columns": X.columns.tolist(),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "created_at": datetime.now().isoformat(),
        "framework_version": "4.0",
        "sklearn_version": __import__("sklearn").__version__,
        "text_columns": [c for c in TEXT_COLUMNS if c in X.columns],
        "finbert_enabled": ENABLE_FINBERT_SENTIMENT,
        "selection_metric_source": "training_cross_validation_only",
        "metrics": best_holdout_metrics
    }
    save_json(metadata, output_dir / "run_metadata.json")
    save_json({"feature_columns": X.columns.tolist()}, output_dir / "feature_schema.json")

    banner("FINAL RESULT")
    print("Selected model:", best_model_name)
    print("Holdout metrics:")
    for key, value in best_holdout_metrics.items():
        if key not in ("model", "status"):
            print(f"  {key}: {value}")
    print(f"Saved best pipeline: {output_dir / 'best_model.joblib'}")
    return best_model


def add_finbert_sentiment(df):
    """Optionally append FinBERT sentiment scores to a text column."""
    if not ENABLE_FINBERT_SENTIMENT:
        return df
    if FINBERT_TEXT_COLUMN not in df.columns:
        raise KeyError(
            f"FINBERT_TEXT_COLUMN '{FINBERT_TEXT_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    except ImportError as exc:
        raise ImportError(
            "FinBERT requires optional dependencies. Install with: "
            "pip install -U transformers torch"
        ) from exc

    banner("FINBERT SENTIMENT ENRICHMENT")
    print(f"Loading {FINBERT_MODEL_NAME}; first run may download model weights.")
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    sentiment_pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        device=device, top_k=None
    )
    texts = df[FINBERT_TEXT_COLUMN].fillna("").astype(str).tolist()
    scores_out = []
    for start in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = texts[start:start + FINBERT_BATCH_SIZE]
        raw = sentiment_pipe(batch, truncation=True, max_length=512)
        for item in raw:
            if isinstance(item, dict):
                item = [item]
            scores = {str(x["label"]).lower(): float(x["score"]) for x in item}
            scores_out.append(scores)
    df = df.copy()
    for label in ("positive", "negative", "neutral"):
        df[f"finbert_{label}"] = [row.get(label, np.nan) for row in scores_out]
    df["finbert_label"] = [
        max(("positive", "negative", "neutral"), key=lambda label: row.get(label, -1.0))
        for row in scores_out
    ]
    print("FinBERT sentiment columns added. This is sentiment analysis, not market prediction.")
    return df


# =====================================================================
# 5. UNSUPERVISED ANALYSIS
# =====================================================================

def run_unsupervised(df, output_dir):
    banner("UNSUPERVISED ANALYSIS")
    X = df.copy()

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print("Dropped constant columns:", constant_cols)

    if len(X) < 4:
        print("Unsupervised analysis needs at least 4 rows.")
        return
    if len(X) > MAX_UNSUPERVISED_ROWS:
        print(
            f"Unsupervised analysis skipped: {len(X)} rows exceeds "
            f"MAX_UNSUPERVISED_ROWS={MAX_UNSUPERVISED_ROWS}."
        )
        return

    prep = make_preprocessor(X, dense=True, scale_numeric=True)
    try:
        transformed = prep.fit_transform(X)
        transformed = np.asarray(transformed, dtype=float)
        if transformed.shape[1] > MAX_DENSE_FEATURES:
            raise MemoryError(
                f"Dense feature count {transformed.shape[1]} exceeds "
                f"MAX_DENSE_FEATURES={MAX_DENSE_FEATURES}. Reduce high-cardinality columns."
            )
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        print("Preprocessing for unsupervised models failed:", exc)
        return

    if RUN_CLUSTERING:
        cluster_models = {
            "KMeans_2": KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10),
            "KMeans_3": KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10),
            "Agglomerative_2": AgglomerativeClustering(n_clusters=2),
            "Agglomerative_3": AgglomerativeClustering(n_clusters=3),
            "Birch_3": Birch(n_clusters=3),
            "GaussianMixture_3": GaussianMixture(n_components=3, random_state=RANDOM_STATE),
            "DBSCAN": DBSCAN(),
            "OPTICS": OPTICS(min_samples=max(2, min(10, len(X) // 10))),
        }
        if len(X) <= 1500:
            cluster_models["Spectral_2"] = SpectralClustering(
                n_clusters=2, random_state=RANDOM_STATE, assign_labels="kmeans"
            )
        cluster_rows = []

        for name, model in cluster_models.items():
            try:
                labels = model.fit_predict(transformed)
                labels = np.asarray(labels)
                mask = labels != -1
                clean_labels = labels[mask]
                clean_data = transformed[mask]
                distinct = len(set(clean_labels))
                row = {
                    "model": name,
                    "clusters_excluding_noise": distinct,
                    "noise_points": int(np.sum(labels == -1)),
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "status": "OK"
                }
                if 2 <= distinct < len(clean_labels):
                    row["silhouette"] = float(silhouette_score(clean_data, clean_labels))
                    row["calinski_harabasz"] = float(
                        calinski_harabasz_score(clean_data, clean_labels)
                    )
                    row["davies_bouldin"] = float(
                        davies_bouldin_score(clean_data, clean_labels)
                    )
                cluster_rows.append(row)
                pd.DataFrame({
                    "row_index": X.index,
                    "cluster": labels
                }).to_csv(output_dir / f"clusters_{safe_name(name)}.csv", index=False)
                print(
                    f"{name:24s} clusters={distinct} "
                    f"silhouette={row['silhouette']}"
                )
            except Exception as exc:
                print(f"{name} FAILED: {exc}")
                cluster_rows.append({"model": name, "status": f"FAILED: {exc}"})

        save_table(cluster_rows, output_dir / "clustering_results.csv")

    if RUN_DIMENSIONALITY_REDUCTION:
        max_components = min(3, transformed.shape[0] - 1, transformed.shape[1])
        if max_components >= 1:
            reducers = {
                "PCA": PCA(n_components=max_components, random_state=RANDOM_STATE),
                "FastICA": FastICA(
                    n_components=max_components, random_state=RANDOM_STATE,
                    max_iter=1000, tol=0.01
                ),
            }
            for name, reducer in reducers.items():
                try:
                    values = reducer.fit_transform(transformed)
                    pd.DataFrame(
                        values, index=X.index,
                        columns=[f"{name}_{i+1}" for i in range(values.shape[1])]
                    ).to_csv(output_dir / f"{safe_name(name)}_coordinates.csv")
                    if name == "PCA":
                        save_json(
                            {"explained_variance_ratio": reducer.explained_variance_ratio_.tolist()},
                            output_dir / "PCA_variance.json"
                        )
                    print(name, "transformation saved.")
                except Exception as exc:
                    print(name, "FAILED:", exc)

            if transformed.shape[1] > 1:
                try:
                    components = min(max_components, transformed.shape[1])
                    svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
                    values = svd.fit_transform(transformed)
                    pd.DataFrame(
                        values, index=X.index,
                        columns=[f"SVD_{i+1}" for i in range(values.shape[1])]
                    ).to_csv(output_dir / "SVD_coordinates.csv")
                    print("TruncatedSVD transformation saved.")
                except Exception as exc:
                    print("TruncatedSVD FAILED:", exc)

                if np.min(transformed) >= 0:
                    try:
                        nmf = NMF(n_components=components, init="nndsvda",
                                  random_state=RANDOM_STATE, max_iter=500)
                        values = nmf.fit_transform(transformed)
                        pd.DataFrame(
                            values, index=X.index,
                            columns=[f"NMF_{i+1}" for i in range(values.shape[1])]
                        ).to_csv(output_dir / "NMF_coordinates.csv")
                        print("NMF transformation saved.")
                    except Exception as exc:
                        print("NMF FAILED:", exc)

    if RUN_ANOMALY_DETECTION:
        anomaly_models = {
            "IsolationForest": IsolationForest(random_state=RANDOM_STATE),
            "LocalOutlierFactor": LocalOutlierFactor(contamination="auto"),
            "OneClassSVM": OneClassSVM()
        }
        anomaly_rows = []
        for name, model in anomaly_models.items():
            try:
                labels = model.fit_predict(transformed)
                count = int(np.sum(labels == -1))
                anomaly_rows.append({
                    "model": name, "anomaly_count": count,
                    "anomaly_fraction": count / len(labels), "status": "OK"
                })
                pd.DataFrame({
                    "row_index": X.index,
                    "prediction": labels,
                    "is_anomaly": labels == -1
                }).to_csv(output_dir / f"anomalies_{name}.csv", index=False)
                print(f"{name}: flagged {count} possible anomalies.")
            except Exception as exc:
                print(name, "FAILED:", exc)
                anomaly_rows.append({"model": name, "status": f"FAILED: {exc}"})
        save_table(anomaly_rows, output_dir / "anomaly_results.csv")


# =====================================================================
# 6. PREDICT ON NEW DATA
# =====================================================================

def predict_new_data(output_dir):
    model_path = Path(SAVED_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path.resolve()}. Train a model first."
        )
    model = joblib.load(model_path)
    new_data = load_table(NEW_DATA_PATH)
    validate_educational_dataset(new_data, NEW_DATA_PATH)
    if ENABLE_FINBERT_SENTIMENT:
        new_data = add_finbert_sentiment(new_data)
    for text_column in TEXT_COLUMNS:
        if text_column in new_data.columns:
            new_data[text_column] = new_data[text_column].fillna("").astype(str)

    metadata_path = model_path.parent / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = metadata.get("feature_columns", [])
        if feature_columns:
            missing = [c for c in feature_columns if c not in new_data.columns]
            extra = [c for c in new_data.columns if c not in feature_columns]
            if missing:
                raise ValueError(f"Prediction data is missing required features: {missing}")
            if extra:
                print("Ignoring extra input columns:", extra)
            new_data = new_data.reindex(columns=feature_columns)

    predictions = model.predict(new_data)
    output = new_data.copy()
    output["prediction"] = predictions

    # Include class probabilities when the model supports them.
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(new_data)
            classes = model.named_steps["model"].classes_
            for index, class_name in enumerate(classes):
                output[f"probability_{safe_name(class_name)}"] = probabilities[:, index]
        except Exception as exc:
            print("Could not add class probabilities:", exc)

    destination = Path(PREDICTION_OUTPUT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print("Predictions saved to:", destination.resolve())


# =====================================================================
# 7. MAIN
# =====================================================================

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("CLASSICAL ML AUTO-FRAMEWORK V4")
    print("Output directory:", output_dir.resolve())

    if PREDICT_MODE:
        predict_new_data(output_dir)
        return

    df = load_table(DATA_PATH)
    validate_educational_dataset(df, DATA_PATH)
    if ENABLE_FINBERT_SENTIMENT:
        df = add_finbert_sentiment(df)
        # Include generated FinBERT columns as regular categorical/numeric features.
        # The original text column can also be listed in TEXT_COLUMNS for TF-IDF.
    missing_text_columns = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing_text_columns:
        raise KeyError(f"TEXT_COLUMNS not found in dataset: {missing_text_columns}")
    if TASK.strip().lower() == "unsupervised" and TEXT_COLUMNS:
        raise ValueError("For memory safety, V4 TF-IDF text support is currently for supervised tasks.")
    save_data_audit(df, output_dir)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    task = TASK.strip().lower()
    if task not in ("classification", "regression", "unsupervised"):
        raise ValueError(
            "TASK must be explicitly set to 'classification', 'regression', or 'unsupervised'."
        )

    if task in ("classification", "regression"):
        run_supervised(df, output_dir, task)
        if RUN_UNSUPERVISED_EXTRAS:
            unsupervised_df = df.drop(columns=[TARGET_COLUMN])
            run_unsupervised(unsupervised_df, output_dir)
    else:
        run_unsupervised(df, output_dir)

    banner("RUN COMPLETED")
    print("Review the CSV reports and run_metadata.json in:", output_dir.resolve())
    print("Best fitted pipeline (supervised):", output_dir / "best_model.joblib")
    print("To predict on new rows, set PREDICT_MODE=True and update NEW_DATA_PATH.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        banner("FRAMEWORK STOPPED WITH AN ERROR")
        print(type(error).__name__ + ":", error)
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nFix the reported issue and rerun. No successful output files are deleted.")
