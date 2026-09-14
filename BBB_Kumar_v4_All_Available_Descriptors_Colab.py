# ==========================================================
# Kumar v3 - All Available Descriptor Experiment
# Google Colab ready
# ==========================================================

import sys, subprocess, importlib.util

packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "xgboost": "xgboost",
    "openpyxl": "openpyxl",
}
for module, package in packages.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor

RANDOM_STATE = 42
N_SPLITS = 5

# ---- Upload workbook in Colab ----
try:
    from google.colab import files
    print("Upload Inorganic_Nanoparticle_Brain_Uptake_Kumar_Audited_v3.xlsx")
    uploaded = files.upload()
    INPUT_FILE = list(uploaded.keys())[0]
except ImportError:
    INPUT_FILE = "Inorganic_Nanoparticle_Brain_Uptake_Kumar_Audited_v3.xlsx"

# Use Accepted_Detailed because it contains all available Kumar descriptors.
df = pd.read_excel(INPUT_FILE, sheet_name="Accepted_Detailed")

# Numeric conversion
for c in ["Time_h", "Size_nm", "Brain_uptake_percentID_g", "PEG_MW_Da_if_numeric"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Age_or_weight_g"] = pd.to_numeric(
    df["Age_or_weight_raw"].replace({"NA": np.nan, "N/A": np.nan, "": np.nan}),
    errors="coerce",
)

# Feature engineering
df["Log1p_Time_h"] = np.log1p(df["Time_h"])
df["Log1p_Size_nm"] = np.log1p(df["Size_nm"])

df["Ligand_present"] = np.where(
    df["Surface_modifier_standardized"].eq("None reported"), "No", "Yes"
)
df["Charge_reported"] = np.where(
    df["Charge_standardized"].eq("Not reported"), "No", "Yes"
)
df["Age_reported"] = np.where(df["Age_or_weight_g"].isna(), "No", "Yes")
df["PEG_present"] = df["PEG_status"].astype(str)

# Interaction categories
df["Material_x_Charge"] = (
    df["Material"].astype(str) + "|" + df["Charge_standardized"].astype(str)
)
df["Material_x_Surface"] = (
    df["Material"].astype(str) + "|" + df["Surface_modifier_standardized"].astype(str)
)
df["Material_x_TimeSegment"] = (
    df["Material"].astype(str) + "|" + df["Time_segment"].astype(str)
)
df["Material_x_SizeBin"] = (
    df["Material"].astype(str) + "|" + df["Size_bin"].astype(str)
)

TARGET = "Brain_uptake_percentID_g"
GROUP = "Study_Group"

y = df[TARGET].astype(float)
groups = df[GROUP].astype(str)

# Important:
# Species, organ and administration route are NOT used as predictors here
# because in this Kumar brain subset they are effectively constant
# (mouse, brain, intravenous) and therefore add no predictive information.
# IDs, PMID/Study_Group, Formulation_Group, source URL and formulation name
# are never used as predictors because they can cause leakage / memorization.

FEATURE_SETS = {
    "Basic_old": [
        "Material",
        "Size_nm",
        "Shape_standardized",
        "Charge_standardized",
    ],

    "All_Kumar": [
        "Time_h",
        "Age_or_weight_g",
        "Strain_standardized",
        "Size_nm",
        "Analysis_method_standardized",
        "Material",
        "Shape_standardized",
        "Surface_modifier_standardized",
        "Charge_standardized",
        "PEG_present",
        "PEG_MW_Da_if_numeric",
    ],

    "All_plus_engineering": [
        "Time_h",
        "Log1p_Time_h",
        "Age_or_weight_g",
        "Age_reported",
        "Strain_standardized",
        "Size_nm",
        "Log1p_Size_nm",
        "Size_bin",
        "Analysis_method_standardized",
        "Material",
        "Shape_standardized",
        "Surface_modifier_standardized",
        "Ligand_present",
        "Charge_standardized",
        "Charge_reported",
        "PEG_present",
        "PEG_MW_Da_if_numeric",
        "Time_segment",
        "Material_x_Charge",
        "Material_x_Surface",
        "Material_x_TimeSegment",
        "Material_x_SizeBin",
    ],
}

MODELS = {
    "Linear": (LinearRegression(), True),
    "Ridge": (Ridge(alpha=1.0), True),
    "Lasso": (Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_STATE), True),
    "ElasticNet": (
        ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        True,
    ),
    "SVR": (SVR(kernel="rbf", C=10, epsilon=0.1), True),
    "kNN": (KNeighborsRegressor(n_neighbors=2), True),
    "DecisionTree": (
        DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
        False,
    ),
    "RandomForest": (
        RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    "ExtraTrees": (
        ExtraTreesRegressor(
            n_estimators=500,
            max_depth=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        False,
    ),
    "XGBoost": (
        XGBRegressor(
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
        False,
    ),
}

def make_preprocessor(X, scale_numeric):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical),
    ])

def evaluate(features, model, scale_numeric, log_target=True):
    X = df[features].copy()
    cv = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(df), np.nan)

    for train_idx, test_idx in cv.split(X, y, groups):
        pipe = Pipeline([
            ("preprocess", make_preprocessor(X.iloc[train_idx], scale_numeric)),
            ("model", clone(model)),
        ])

        y_train = y.iloc[train_idx]
        if log_target:
            y_train = np.log1p(y_train)

        pipe.fit(X.iloc[train_idx], y_train)

        pred = pipe.predict(X.iloc[test_idx])
        if log_target:
            pred = np.expm1(np.clip(pred, -20, 20))

        # Uptake cannot be negative on the original scale.
        pred = np.maximum(pred, 0)
        oof[test_idx] = pred

    return {
        "MAE": mean_absolute_error(y, oof),
        "RMSE": np.sqrt(mean_squared_error(y, oof)),
        "R2": r2_score(y, oof),
        "Spearman": spearmanr(y, oof).statistic,
    }

results = []

for feature_set_name, features in FEATURE_SETS.items():
    for model_name, (model, scale_numeric) in MODELS.items():
        metrics = evaluate(
            features,
            model,
            scale_numeric,
            log_target=True,
        )
        results.append({
            "Feature_Set": feature_set_name,
            "Model": model_name,
            **metrics,
        })

results = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

print("\nTOP RESULTS")
print(results.head(15).to_string(index=False))

results.to_csv("Kumar_all_descriptors_model_comparison.csv", index=False)
results.to_excel("Kumar_all_descriptors_model_comparison.xlsx", index=False)

try:
    from google.colab import files
    files.download("Kumar_all_descriptors_model_comparison.xlsx")
except ImportError:
    pass
