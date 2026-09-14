# ================================================================
# MSc THESIS - INORGANIC NANOPARTICLE BRAIN UPTAKE ML PIPELINE
# Google Colab ready
#
# Expected input workbook:
# Inorganic_Nanoparticle_Brain_Uptake_Kumar_Audited_v3.xlsx
# Sheet: Final_ML_Input
#
# Author workflow for:
# Moamen Elsayed - M.Sc. Digital Chemistry, University of Gdansk
#
# IMPORTANT SCIENTIFIC NOTE:
# The target is quantitative brain biodistribution / brain uptake.
# It is NOT automatically proof of BBB penetration or parenchymal uptake.
# ================================================================

# -----------------------------
# 0. Install required packages
# -----------------------------
import sys
import subprocess
import importlib.util

required_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
    "xgboost": "xgboost",
    "shap": "shap",
}

for module_name, package_name in required_packages.items():
    if importlib.util.find_spec(module_name) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name]
        )

# -----------------------------
# 1. Imports
# -----------------------------
import os
import json
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
USE_LOG_TARGET = True
OUTPUT_DIR = Path("BBB_ML_Kumar_v3_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Packages loaded successfully.")


# -----------------------------
# 2. Upload / locate workbook
# -----------------------------
def choose_input_file():
    """
    In Google Colab: opens file upload dialog.
    Outside Colab: looks for the expected workbook in current directory.
    """
    try:
        from google.colab import files
        print("\nUpload the audited v3 Excel workbook now:")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No file was uploaded.")
        xlsx_files = [name for name in uploaded.keys() if name.lower().endswith(".xlsx")]
        if not xlsx_files:
            raise RuntimeError("Please upload an .xlsx file.")
        # If multiple files are uploaded, prefer the expected v3 name.
        preferred = [
            f for f in xlsx_files
            if "Audited_v3" in f or "audited_v3" in f.lower()
        ]
        return preferred[0] if preferred else xlsx_files[0]
    except ImportError:
        expected = "Inorganic_Nanoparticle_Brain_Uptake_Kumar_Audited_v3.xlsx"
        if Path(expected).exists():
            return expected
        candidates = list(Path(".").glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(
                "No .xlsx file found. Place the audited v3 workbook in the working directory."
            )
        return str(candidates[0])

INPUT_FILE = choose_input_file()
SHEET_NAME = "Final_ML_Input"

print(f"\nInput file: {INPUT_FILE}")
print(f"Input sheet: {SHEET_NAME}")


# -----------------------------
# 3. Load and validate data
# -----------------------------
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

required_columns = [
    "Observation_ID",
    "Study_Group",
    "Formulation_Group",
    "Time_h",
    "Material",
    "Size_nm",
    "Shape",
    "Surface_modifier",
    "Charge",
    "PEG_status",
    "PEG_MW_Da_if_numeric",
    "Analysis_method",
    "Strain",
    "Brain_uptake_percentID_g",
    "Source_URL",
    "QC_Status",
    "QC_Flags",
]

missing_cols = [c for c in required_columns if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("\nInitial dataset shape:", df.shape)

# Keep accepted records only.
accepted_mask = df["QC_Status"].astype(str).str.startswith("Accepted", na=False)
df = df.loc[accepted_mask].copy()

# Force numeric target and essential numeric columns.
for col in ["Time_h", "Size_nm", "PEG_MW_Da_if_numeric", "Brain_uptake_percentID_g"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove impossible / unusable essential values.
exclusion_reasons = []

def exclude_with_reason(frame, mask, reason):
    if mask.any():
        temp = frame.loc[mask].copy()
        temp["ML_Exclusion_Reason"] = reason
        exclusion_reasons.append(temp)
    return frame.loc[~mask].copy()

df = exclude_with_reason(
    df,
    df["Brain_uptake_percentID_g"].isna(),
    "Missing target value"
)
df = exclude_with_reason(
    df,
    df["Brain_uptake_percentID_g"] < 0,
    "Negative target value"
)
df = exclude_with_reason(
    df,
    df["Study_Group"].isna() | (df["Study_Group"].astype(str).str.strip() == ""),
    "Missing Study_Group"
)

# Exact duplicate audit across scientifically meaningful fields.
duplicate_key = [
    "Study_Group",
    "Formulation_Group",
    "Time_h",
    "Material",
    "Size_nm",
    "Shape",
    "Surface_modifier",
    "Charge",
    "PEG_status",
    "PEG_MW_Da_if_numeric",
    "Analysis_method",
    "Strain",
    "Brain_uptake_percentID_g",
]
dup_mask = df.duplicated(subset=duplicate_key, keep="first")
df = exclude_with_reason(df, dup_mask, "Exact duplicate row")

df = df.reset_index(drop=True)

excluded_df = (
    pd.concat(exclusion_reasons, ignore_index=True)
    if exclusion_reasons
    else pd.DataFrame(columns=list(df.columns) + ["ML_Exclusion_Reason"])
)

print("Final modeling rows:", len(df))
print("Unique study groups:", df["Study_Group"].nunique())
print("Unique formulations:", df["Formulation_Group"].nunique())
print("\nMaterial counts:")
print(df["Material"].value_counts(dropna=False))

if df["Study_Group"].nunique() < N_SPLITS:
    N_SPLITS = df["Study_Group"].nunique()
    print(f"\nN_SPLITS reduced to {N_SPLITS} because of limited study groups.")

if N_SPLITS < 2:
    raise ValueError("At least 2 independent Study_Group values are required.")

# Target
TARGET = "Brain_uptake_percentID_g"
GROUP_COL = "Study_Group"

y = df[TARGET].astype(float).copy()
groups = df[GROUP_COL].astype(str).copy()

if (y < 0).any():
    raise ValueError("Target contains negative values, which is incompatible with log1p.")

print("\nTarget summary:")
print(y.describe())


# -----------------------------
# 4. Missingness report
# -----------------------------
missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_n": [df[c].isna().sum() for c in df.columns],
    "Missing_percent": [100 * df[c].isna().mean() for c in df.columns],
})
missing_report = missing_report.sort_values(
    ["Missing_percent", "Missing_n"], ascending=False
)

missing_report.to_csv(
    OUTPUT_DIR / "01_missingness_report.csv", index=False
)


# -----------------------------
# 5. Feature sets
# -----------------------------
# A = basic physicochemical
FEATURE_SET_A = [
    "Material",
    "Size_nm",
    "Shape",
    "Charge",
]

# B = basic + surface descriptors
FEATURE_SET_B = FEATURE_SET_A + [
    "Surface_modifier",
    "PEG_status",
    "PEG_MW_Da_if_numeric",
]

# C = basic + surface + experimental context
FEATURE_SET_C = FEATURE_SET_B + [
    "Time_h",
    "Analysis_method",
    "Strain",
]

FEATURE_SETS = {
    "A_basic_physicochemical": FEATURE_SET_A,
    "B_plus_surface": FEATURE_SET_B,
    "C_plus_experimental_context": FEATURE_SET_C,
}

# Ensure all features exist.
for name, cols in FEATURE_SETS.items():
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

print("\nFeature sets:")
for name, cols in FEATURE_SETS.items():
    print(name, "->", cols)


# -----------------------------
# 6. Preprocessing builders
# -----------------------------
def make_preprocessor(X, scale_numeric=True):
    numeric_features = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_features = [
        c for c in X.columns
        if c not in numeric_features
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_features, categorical_features


# -----------------------------
# 7. Models
# -----------------------------
# These settings match the conservative settings documented in the thesis draft.
MODEL_SPECS = {
    "Dummy_Mean": {
        "model": DummyRegressor(strategy="mean"),
        "scale": False,
    },
    "Linear_Regression": {
        "model": LinearRegression(),
        "scale": True,
    },
    "Ridge": {
        "model": Ridge(alpha=1.0),
        "scale": True,
    },
    "Lasso": {
        "model": Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE),
        "scale": True,
    },
    "SVR_RBF": {
        "model": SVR(kernel="rbf", C=10, epsilon=0.1),
        "scale": True,
    },
    "kNN": {
        "model": KNeighborsRegressor(n_neighbors=2, weights="uniform"),
        "scale": True,
    },
    "Decision_Tree": {
        "model": DecisionTreeRegressor(
            max_depth=3,
            random_state=RANDOM_STATE
        ),
        "scale": False,
    },
    "Random_Forest": {
        "model": RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale": False,
    },
    "Gradient_Boosting": {
        "model": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        "scale": False,
    },
    "XGBoost": {
        "model": XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "scale": False,
    },
}


# -----------------------------
# 8. Metrics
# -----------------------------
def q2_score(y_true, y_pred):
    """
    Cross-validated Q2 defined as:
    Q2 = 1 - PRESS / TSS
    using pooled out-of-fold predictions and the overall observed mean.

    Under this exact definition, pooled Q2 is mathematically equivalent
    to pooled out-of-fold R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    press = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    if tss == 0:
        return np.nan
    return 1.0 - press / tss


def safe_spearman(y_true, y_pred):
    try:
        rho, p = spearmanr(y_true, y_pred)
        return float(rho), float(p)
    except Exception:
        return np.nan, np.nan


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    q2 = q2_score(y_true, y_pred)
    rho, rho_p = safe_spearman(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2_OOF": r2,
        "Q2_OOF": q2,
        "Spearman_rho": rho,
        "Spearman_p": rho_p,
    }


# -----------------------------
# 9. Grouped CV evaluation
# -----------------------------
def transform_target(y_train):
    if USE_LOG_TARGET:
        return np.log1p(y_train)
    return np.asarray(y_train, dtype=float)


def inverse_target(y_pred_transformed):
    y_pred_transformed = np.asarray(y_pred_transformed, dtype=float)

    if USE_LOG_TARGET:
        # Keep numerical computation stable but do not artificially
        # improve predictions. Final physical target is clipped at 0.
        y_pred_transformed = np.clip(y_pred_transformed, -20, 20)
        y_pred = np.expm1(y_pred_transformed)
    else:
        y_pred = y_pred_transformed

    return np.maximum(y_pred, 0.0)


def run_grouped_cv(X, y, groups, model, scale_numeric, config_name):
    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_records = []
    fitted_folds = []

    for fold_id, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups), start=1
    ):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        preprocessor, num_cols, cat_cols = make_preprocessor(
            X_train, scale_numeric=scale_numeric
        )

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", clone(model)),
        ])

        y_train_fit = transform_target(y_train)
        pipeline.fit(X_train, y_train_fit)

        pred_transformed = pipeline.predict(X_test)
        y_pred = inverse_target(pred_transformed)

        oof_pred[test_idx] = y_pred

        fold_metrics = calculate_metrics(y_test.values, y_pred)

        fold_records.append({
            "Configuration": config_name,
            "Fold": fold_id,
            "Train_rows": len(train_idx),
            "Test_rows": len(test_idx),
            "Train_groups": len(set(groups.iloc[train_idx])),
            "Test_groups": len(set(groups.iloc[test_idx])),
            "Test_group_names": " | ".join(sorted(set(groups.iloc[test_idx]))),
            **fold_metrics,
        })

        fitted_folds.append({
            "fold": fold_id,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "pipeline": pipeline,
        })

    if np.isnan(oof_pred).any():
        raise RuntimeError(f"Missing OOF predictions in {config_name}")

    pooled_metrics = calculate_metrics(y.values, oof_pred)

    return oof_pred, pd.DataFrame(fold_records), fitted_folds, pooled_metrics


# -----------------------------
# 10. Run all configurations
# -----------------------------
comparison_rows = []
all_predictions = []
all_fold_metrics = []
fitted_objects = {}

total_configs = len(FEATURE_SETS) * len(MODEL_SPECS)
config_counter = 0

for feature_set_name, feature_cols in FEATURE_SETS.items():
    X = df[feature_cols].copy()

    for model_name, spec in MODEL_SPECS.items():
        config_counter += 1
        config_name = f"{model_name}__{feature_set_name}"

        print(
            f"[{config_counter}/{total_configs}] "
            f"Running {config_name}"
        )

        try:
            oof_pred, fold_df, fitted_folds, metrics = run_grouped_cv(
                X=X,
                y=y,
                groups=groups,
                model=spec["model"],
                scale_numeric=spec["scale"],
                config_name=config_name,
            )

            comparison_rows.append({
                "Model": model_name,
                "Feature_Set": feature_set_name,
                "N_rows": len(df),
                "N_study_groups": groups.nunique(),
                "N_folds": N_SPLITS,
                "Target_transform": "log1p" if USE_LOG_TARGET else "none",
                **metrics,
            })

            pred_df = pd.DataFrame({
                "Observation_ID": df["Observation_ID"].values,
                "Study_Group": df["Study_Group"].values,
                "Formulation_Group": df["Formulation_Group"].values,
                "Material": df["Material"].values,
                "Time_h": df["Time_h"].values,
                "Observed": y.values,
                "Predicted": oof_pred,
                "Residual": y.values - oof_pred,
                "Model": model_name,
                "Feature_Set": feature_set_name,
            })

            all_predictions.append(pred_df)
            all_fold_metrics.append(fold_df)
            fitted_objects[(model_name, feature_set_name)] = fitted_folds

        except Exception as exc:
            print(f"FAILED: {config_name}")
            print("Reason:", repr(exc))

            comparison_rows.append({
                "Model": model_name,
                "Feature_Set": feature_set_name,
                "N_rows": len(df),
                "N_study_groups": groups.nunique(),
                "N_folds": N_SPLITS,
                "Target_transform": "log1p" if USE_LOG_TARGET else "none",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2_OOF": np.nan,
                "Q2_OOF": np.nan,
                "Spearman_rho": np.nan,
                "Spearman_p": np.nan,
                "Failure": repr(exc),
            })

comparison_df = pd.DataFrame(comparison_rows)
predictions_df = pd.concat(all_predictions, ignore_index=True)
fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)

# Rank NON-DUMMY models by MAE.
comparison_df["Is_Dummy"] = comparison_df["Model"].eq("Dummy_Mean")
comparison_df["Rank_MAE_non_dummy"] = np.nan

non_dummy_idx = comparison_df.index[
    (~comparison_df["Is_Dummy"]) & comparison_df["MAE"].notna()
]
comparison_df.loc[
    non_dummy_idx, "Rank_MAE_non_dummy"
] = comparison_df.loc[non_dummy_idx, "MAE"].rank(method="min")

comparison_df = comparison_df.sort_values(
    ["Is_Dummy", "MAE"], ascending=[True, True]
).reset_index(drop=True)

comparison_df.to_csv(
    OUTPUT_DIR / "02_model_comparison.csv", index=False
)
predictions_df.to_csv(
    OUTPUT_DIR / "03_all_oof_predictions.csv", index=False
)
fold_metrics_df.to_csv(
    OUTPUT_DIR / "04_fold_metrics.csv", index=False
)

print("\nTop model configurations:")
print(
    comparison_df[
        [
            "Model", "Feature_Set", "MAE", "RMSE",
            "R2_OOF", "Q2_OOF", "Spearman_rho"
        ]
    ].head(12).to_string(index=False)
)


# -----------------------------
# 11. Identify best non-dummy configuration
# -----------------------------
valid_non_dummy = comparison_df[
    (~comparison_df["Is_Dummy"]) &
    comparison_df["MAE"].notna()
].copy()

if valid_non_dummy.empty:
    raise RuntimeError("No non-dummy model completed successfully.")

best_row = valid_non_dummy.sort_values("MAE").iloc[0]
BEST_MODEL = best_row["Model"]
BEST_FEATURE_SET = best_row["Feature_Set"]

print("\nBEST NON-DUMMY CONFIGURATION BY MAE")
print("-----------------------------------")
print("Model:", BEST_MODEL)
print("Feature set:", BEST_FEATURE_SET)
print("MAE:", best_row["MAE"])
print("RMSE:", best_row["RMSE"])
print("R2_OOF:", best_row["R2_OOF"])
print("Q2_OOF:", best_row["Q2_OOF"])
print("Spearman rho:", best_row["Spearman_rho"])

best_predictions = predictions_df[
    (predictions_df["Model"] == BEST_MODEL) &
    (predictions_df["Feature_Set"] == BEST_FEATURE_SET)
].copy()

best_predictions.to_csv(
    OUTPUT_DIR / "05_best_model_oof_predictions.csv", index=False
)


# -----------------------------
# 12. Compare against mean baseline
# -----------------------------
dummy_rows = comparison_df[
    comparison_df["Model"] == "Dummy_Mean"
].copy()

baseline_comparison = valid_non_dummy[
    ["Model", "Feature_Set", "MAE", "RMSE", "R2_OOF", "Q2_OOF", "Spearman_rho"]
].copy()

dummy_map = dummy_rows.set_index("Feature_Set")["MAE"].to_dict()
baseline_comparison["Dummy_MAE_same_feature_set"] = (
    baseline_comparison["Feature_Set"].map(dummy_map)
)
baseline_comparison["MAE_improvement_vs_dummy"] = (
    baseline_comparison["Dummy_MAE_same_feature_set"] -
    baseline_comparison["MAE"]
)
baseline_comparison["Percent_MAE_improvement_vs_dummy"] = (
    100 *
    baseline_comparison["MAE_improvement_vs_dummy"] /
    baseline_comparison["Dummy_MAE_same_feature_set"]
)

baseline_comparison.to_csv(
    OUTPUT_DIR / "06_baseline_comparison.csv", index=False
)


# -----------------------------
# 13. Fold-wise held-out permutation importance
# -----------------------------
best_features = FEATURE_SETS[BEST_FEATURE_SET]
best_spec = MODEL_SPECS[BEST_MODEL]
best_X = df[best_features].copy()

best_folds = fitted_objects[(BEST_MODEL, BEST_FEATURE_SET)]
importance_records = []

for fold_info in best_folds:
    fold_id = fold_info["fold"]
    test_idx = fold_info["test_idx"]
    pipeline = fold_info["pipeline"]

    X_test = best_X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    # Custom scorer converts transformed predictions back to the
    # original %ID/g scale before calculating held-out MAE.
    def original_scale_neg_mae(estimator, X_eval, y_eval):
        pred_t = estimator.predict(X_eval)
        pred_original = inverse_target(pred_t)
        return -mean_absolute_error(y_eval, pred_original)

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=original_scale_neg_mae,
        n_repeats=10,
        random_state=RANDOM_STATE + fold_id,
        n_jobs=1,
    )

    for feature, mean_imp, std_imp in zip(
        X_test.columns,
        result.importances_mean,
        result.importances_std,
    ):
        importance_records.append({
            "Fold": fold_id,
            "Feature": feature,
            "Importance_MAE_mean": mean_imp,
            "Importance_MAE_std": std_imp,
        })

perm_fold_df = pd.DataFrame(importance_records)

perm_summary_df = (
    perm_fold_df
    .groupby("Feature", as_index=False)
    .agg(
        Mean_importance=("Importance_MAE_mean", "mean"),
        SD_across_folds=("Importance_MAE_mean", "std"),
        Median_importance=("Importance_MAE_mean", "median"),
        Positive_folds=("Importance_MAE_mean", lambda s: int((s > 0).sum())),
        N_folds=("Fold", "nunique"),
    )
    .sort_values("Mean_importance", ascending=False)
)

perm_fold_df.to_csv(
    OUTPUT_DIR / "07_permutation_importance_by_fold.csv", index=False
)
perm_summary_df.to_csv(
    OUTPUT_DIR / "08_permutation_importance_summary.csv", index=False
)


# -----------------------------
# 14. Fit best configuration on all data
#     FOR EXPLORATORY INTERPRETATION ONLY
# -----------------------------
best_preprocessor, _, _ = make_preprocessor(
    best_X, scale_numeric=best_spec["scale"]
)

best_full_pipeline = Pipeline([
    ("preprocess", best_preprocessor),
    ("model", clone(best_spec["model"])),
])

best_full_pipeline.fit(
    best_X,
    transform_target(y)
)

# Full-data fitted predictions are NOT validation results.
full_fit_pred = inverse_target(best_full_pipeline.predict(best_X))

full_fit_metrics = calculate_metrics(y.values, full_fit_pred)
pd.DataFrame([{
    "Model": BEST_MODEL,
    "Feature_Set": BEST_FEATURE_SET,
    "WARNING": "Training/full-fit metrics only; do not use as evidence of generalization.",
    **full_fit_metrics
}]).to_csv(
    OUTPUT_DIR / "09_full_fit_metrics_NOT_VALIDATION.csv",
    index=False
)


# -----------------------------
# 15. Exploratory SHAP
# -----------------------------
shap_status = "Not attempted"

try:
    import shap

    tree_models = {
        "Decision_Tree",
        "Random_Forest",
        "Gradient_Boosting",
        "XGBoost",
    }

    if BEST_MODEL in tree_models:
        preprocessor = best_full_pipeline.named_steps["preprocess"]
        fitted_model = best_full_pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(best_X)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_transformed)

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "10_SHAP_summary_exploratory.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close()

        # Aggregate mean absolute SHAP by transformed feature.
        if isinstance(shap_values, list):
            shap_arr = np.asarray(shap_values[0])
        else:
            shap_arr = np.asarray(shap_values)

        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)

        shap_df = pd.DataFrame({
            "Transformed_feature": feature_names,
            "Mean_abs_SHAP": mean_abs_shap,
        }).sort_values("Mean_abs_SHAP", ascending=False)

        shap_df.to_csv(
            OUTPUT_DIR / "11_SHAP_feature_importance_exploratory.csv",
            index=False,
        )

        shap_status = "Completed for best tree-based model"
    else:
        shap_status = (
            f"Skipped because best model ({BEST_MODEL}) is not tree-based."
        )

except Exception as exc:
    shap_status = f"SHAP failed safely: {repr(exc)}"

print("\nSHAP status:", shap_status)


# -----------------------------
# 16. Diagnostic plots
# -----------------------------
# 16a. Target distribution
plt.figure(figsize=(7, 5))
plt.hist(y, bins=20)
plt.xlabel("Brain uptake (%ID/g)")
plt.ylabel("Count")
plt.title("Distribution of Quantitative Brain Uptake")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "12_target_distribution.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

# 16b. Brain uptake by material
materials = list(df["Material"].dropna().unique())
material_data = [
    df.loc[df["Material"] == m, TARGET].values
    for m in materials
]

plt.figure(figsize=(7, 5))
plt.boxplot(material_data, tick_labels=materials)
plt.ylabel("Brain uptake (%ID/g)")
plt.xlabel("Material")
plt.title("Brain Uptake by Inorganic Nanoparticle Material")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "13_brain_uptake_by_material.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

# 16c. Best-model observed vs predicted
obs = best_predictions["Observed"].values
pred = best_predictions["Predicted"].values

plt.figure(figsize=(6, 6))
plt.scatter(obs, pred, alpha=0.75)
min_val = min(np.min(obs), np.min(pred))
max_val = max(np.max(obs), np.max(pred))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Observed brain uptake (%ID/g)")
plt.ylabel("Out-of-fold predicted brain uptake (%ID/g)")
plt.title(f"Observed vs Predicted\n{BEST_MODEL} | {BEST_FEATURE_SET}")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "14_best_model_observed_vs_predicted.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

# 16d. Residual plot
plt.figure(figsize=(7, 5))
plt.scatter(
    best_predictions["Predicted"],
    best_predictions["Residual"],
    alpha=0.75,
)
plt.axhline(0, linestyle="--")
plt.xlabel("Out-of-fold predicted brain uptake (%ID/g)")
plt.ylabel("Residual (Observed - Predicted)")
plt.title("Best Model Out-of-Fold Residuals")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "15_best_model_residuals.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

# 16e. Model comparison by MAE
plot_df = valid_non_dummy.sort_values("MAE").head(15).copy()
labels = (
    plot_df["Model"] + "\n" +
    plot_df["Feature_Set"].str.replace("_", " ", regex=False)
)

plt.figure(figsize=(11, 7))
plt.barh(range(len(plot_df)), plot_df["MAE"])
plt.yticks(range(len(plot_df)), labels)
plt.xlabel("Cross-validated MAE (%ID/g)")
plt.title("Top 15 Model Configurations by Out-of-Fold MAE")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "16_model_comparison_MAE.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close()

# 16f. Permutation importance summary
if not perm_summary_df.empty:
    top_imp = perm_summary_df.head(15).sort_values("Mean_importance")

    plt.figure(figsize=(8, 6))
    plt.barh(top_imp["Feature"], top_imp["Mean_importance"])
    plt.xlabel("Increase in held-out MAE after permutation")
    plt.title("Held-out Permutation Importance")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "17_permutation_importance.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


# -----------------------------
# 17. Dataset descriptive summary
# -----------------------------
material_summary = (
    df.groupby("Material")
    .agg(
        Rows=(TARGET, "size"),
        Study_groups=(GROUP_COL, "nunique"),
        Formulations=("Formulation_Group", "nunique"),
        Mean_brain_uptake=(TARGET, "mean"),
        Median_brain_uptake=(TARGET, "median"),
        Min_brain_uptake=(TARGET, "min"),
        Max_brain_uptake=(TARGET, "max"),
    )
    .reset_index()
)

study_summary = (
    df.groupby(GROUP_COL)
    .agg(
        Rows=(TARGET, "size"),
        Formulations=("Formulation_Group", "nunique"),
        Materials=("Material", lambda s: " | ".join(sorted(set(map(str, s))))),
        Mean_brain_uptake=(TARGET, "mean"),
        Min_time_h=("Time_h", "min"),
        Max_time_h=("Time_h", "max"),
    )
    .reset_index()
)

material_summary.to_csv(
    OUTPUT_DIR / "18_material_summary.csv", index=False
)
study_summary.to_csv(
    OUTPUT_DIR / "19_study_group_summary.csv", index=False
)


# -----------------------------
# 18. Save clean modeling dataset
# -----------------------------
df.to_csv(
    OUTPUT_DIR / "20_final_modeling_dataset.csv", index=False
)

if not excluded_df.empty:
    excluded_df.to_csv(
        OUTPUT_DIR / "21_rows_excluded_during_ML_precheck.csv",
        index=False,
    )


# -----------------------------
# 19. Create Excel results workbook
# -----------------------------
results_xlsx = OUTPUT_DIR / "BBB_ML_Kumar_v3_results.xlsx"

with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
    comparison_df.to_excel(
        writer, sheet_name="Model_Comparison", index=False
    )
    baseline_comparison.to_excel(
        writer, sheet_name="Baseline_Comparison", index=False
    )
    best_predictions.to_excel(
        writer, sheet_name="Best_OOF_Predictions", index=False
    )
    fold_metrics_df.to_excel(
        writer, sheet_name="Fold_Metrics", index=False
    )
    perm_summary_df.to_excel(
        writer, sheet_name="Permutation_Importance", index=False
    )
    missing_report.to_excel(
        writer, sheet_name="Missingness", index=False
    )
    material_summary.to_excel(
        writer, sheet_name="Material_Summary", index=False
    )
    study_summary.to_excel(
        writer, sheet_name="Study_Groups", index=False
    )
    df.to_excel(
        writer, sheet_name="Modeling_Data", index=False
    )

    if not excluded_df.empty:
        excluded_df.to_excel(
            writer, sheet_name="ML_Precheck_Excluded", index=False
        )


# -----------------------------
# 20. Reproducibility summary
# -----------------------------
summary = {
    "input_file": INPUT_FILE,
    "sheet": SHEET_NAME,
    "rows_used": int(len(df)),
    "unique_study_groups": int(df[GROUP_COL].nunique()),
    "unique_formulations": int(df["Formulation_Group"].nunique()),
    "materials": {
        str(k): int(v)
        for k, v in df["Material"].value_counts().to_dict().items()
    },
    "target": TARGET,
    "target_unit": "%ID/g",
    "target_transform": "log1p" if USE_LOG_TARGET else "none",
    "grouping_variable": GROUP_COL,
    "cross_validation": f"GroupKFold(n_splits={N_SPLITS})",
    "random_state": RANDOM_STATE,
    "best_model_by_MAE": BEST_MODEL,
    "best_feature_set_by_MAE": BEST_FEATURE_SET,
    "best_MAE": float(best_row["MAE"]),
    "best_RMSE": float(best_row["RMSE"]),
    "best_R2_OOF": float(best_row["R2_OOF"]),
    "best_Q2_OOF": float(best_row["Q2_OOF"]),
    "best_Spearman_rho": (
        None if pd.isna(best_row["Spearman_rho"])
        else float(best_row["Spearman_rho"])
    ),
    "shap_status": shap_status,
    "scientific_warning": (
        "Brain biodistribution is not automatically equivalent to "
        "BBB penetration or parenchymal localization."
    ),
}

with open(
    OUTPUT_DIR / "22_run_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# -----------------------------
# 21. Human-readable report
# -----------------------------
best_interpretation = (
    "The selected model shows positive out-of-fold R2."
    if best_row["R2_OOF"] > 0
    else (
        "The selected model has a negative out-of-fold R2 and therefore "
        "does not outperform the mean baseline in explained variance."
    )
)

report_text = f"""
MSc THESIS MACHINE-LEARNING RUN SUMMARY
=======================================

Dataset
-------
Rows used: {len(df)}
Unique study groups: {df[GROUP_COL].nunique()}
Unique formulations: {df['Formulation_Group'].nunique()}
Materials:
{df['Material'].value_counts().to_string()}

Target
------
{TARGET}
Unit: %ID/g

Validation
----------
GroupKFold with {N_SPLITS} folds.
Grouping variable: Study_Group.
All preprocessing was fitted inside each training fold.
Target transformation: {'log1p' if USE_LOG_TARGET else 'none'}.

Best non-dummy configuration by MAE
-----------------------------------
Model: {BEST_MODEL}
Feature set: {BEST_FEATURE_SET}
MAE: {best_row['MAE']:.6f}
RMSE: {best_row['RMSE']:.6f}
Out-of-fold R2: {best_row['R2_OOF']:.6f}
Out-of-fold Q2: {best_row['Q2_OOF']:.6f}
Spearman rho: {best_row['Spearman_rho']}

Interpretation
--------------
{best_interpretation}

IMPORTANT:
"Best" means best relative performance among the tested configurations.
It does not automatically mean that the model is scientifically predictive.

Brain biodistribution / brain uptake is not automatically proof of BBB
penetration or parenchymal localization. Mechanistic BBB claims require
verification from the corresponding primary studies.

SHAP
----
{shap_status}
"""

with open(
    OUTPUT_DIR / "23_READ_ME_RESULTS.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(report_text)

print(report_text)


# -----------------------------
# 22. Zip all results
# -----------------------------
zip_base = "BBB_ML_Kumar_v3_results"
zip_path = shutil.make_archive(
    zip_base,
    "zip",
    root_dir=OUTPUT_DIR
)

print("\nAll results saved.")
print("Excel results:", results_xlsx)
print("ZIP results:", zip_path)


# -----------------------------
# 23. Download results in Colab
# -----------------------------
try:
    from google.colab import files

    print("\nDownloading result files...")
    files.download(str(results_xlsx))
    files.download(zip_path)

except ImportError:
    print("\nNot running in Google Colab.")
    print("Files were saved locally in:", OUTPUT_DIR.resolve())
