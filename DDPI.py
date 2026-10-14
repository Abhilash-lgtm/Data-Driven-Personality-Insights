import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

import os

# ------------------------------------
# Load data + standardize column names
# ------------------------------------
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "responses_data.csv")
df = pd.read_csv(csv_file)

# tandardize column names (trim spaces, collapse multiple spaces)
df.columns = [str(c).strip() for c in df.columns]
df.columns = [re.sub(r"\s+", " ", c) for c in df.columns]

# Add an ID if not present
if "ID" not in df.columns:
    df["ID"] = np.arange(len(df))
print("Dataset shape:", df.shape)

# ---------------------------
# Basic cleaning
# ---------------------------
# Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print("Removed duplicates:", before - len(df))

# Ensure known numeric columns are numeric (if they exist)
for col in ["Age", "Height", "Weight", "Number of siblings"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize common categorical columns

# --- Page 2 ---
for col in ["Gender", "Education", "Smoking", "Alcohol", "Village - town", "House - block of flats",
            "Left - right handed", "Punctuality", "Lying", "Internet usage"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip().str.lower()

# ------------------------------------------
# Outlier checks + corrections (Height, Weight)
# ------------------------------------------
# Height is in cm in this dataset (typical 140-210 cm)
# If your dataset uses meters, adjust thresholds accordingly.
if "Height" in df.columns:
    df.loc[(df["Height"] < 120) | (df["Height"] > 230), "Height"] = np.nan

if "Weight" in df.columns:
    df.loc[(df["Weight"] < 35) | (df["Weight"] > 200), "Weight"] = np.nan

if "Age" in df.columns:
    df.loc[(df["Age"] < 14) | (df["Age"] > 60), "Age"] = np.nan

# ------------------------------------------
# Contextual replacements (Height/Weight by Gender+Age averages)
# ------------------------------------------
def fill_by_group_mean(frame, value_col, group_cols):
    """Fill NaN in value_col using group means; if missing group mean, use global mean."""
    if value_col not in frame.columns:
        return frame
    g = frame.groupby(group_cols)[value_col].mean()
    global_mean = frame[value_col].mean()
    def _fill(row):
        if pd.notna(row[value_col]):
            return row[value_col]
        key = tuple(row[c] for c in group_cols)
        if key in g.index and pd.notna(g.loc[key]):
            return g.loc[key]
        return global_mean
    frame[value_col] = frame.apply(_fill, axis=1)
    return frame

# Make a "Age bin" for better demographic averages
if "Age" in df.columns:
    df["age_bin"] = pd.cut(df["Age"], bins=[0, 18, 24, 30, 40, 60], labels=["<=18", "19-24", "25-30", "31-40", "41-60"])

if "Gender" in df.columns and "Age" in df.columns:
    # fill Gender if missing using mode
    if df["Gender"].isna().any():

# --- Page 3 ---
        df["Gender"] = df["Gender"].fillna(df["Gender"].mode().iloc[0])

    # fill Height/Weight by (Gender, age_bin)
    if "Height" in df.columns:
        df = fill_by_group_mean(df, "Height", ["Gender", "age_bin"])
    if "Weight" in df.columns:
        df = fill_by_group_mean(df, "Weight", ["Gender", "age_bin"])

# ------------------------------------------
# Identify Likert-scale columns and validate bounds (1-5)
# ------------------------------------------
# Many survey columns are 1-5 ratings.
# We'll detect such columns by checking if most non-null values are between 1 and 5.
likert_cols = []
for c in df.columns:
    if c in ["ID", "Age", "Height", "Weight", "Number of siblings", "age_bin"]:
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        s = df[c].dropna()
        if len(s) > 0:
            frac_in_range = ((s >= 1) & (s <= 5)).mean()
            if frac_in_range >= 0.95:
                likert_cols.append(c)

# Coerce Likert columns to numeric and clip to 1..5
for c in likert_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].clip(1, 5)

print("Detected Likert columns:", len(likert_cols))

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

# Remove helper age_bin from cat_cols (we will later one-hot it)
if "age_bin" in cat_cols:
    cat_cols.remove("age_bin")

for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# KNN impute Likert columns (and optionally other numeric columns)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# Make sure ID is not imputed as a feature
num_cols_for_impute = [c for c in num_cols if c != "ID"]

# --- Page 4 ---
imputer = KNNImputer(n_neighbors=25, weights="distance")
df[num_cols_for_impute] = imputer.fit_transform(df[num_cols_for_impute])

# Round Likert columns to nearest integer after imputation
for c in likert_cols:
    df[c] = np.round(df[c]).astype(int)

# ------------------------------------------
# Feature Engineering
cols = list(df.columns)
personality_block = cols[76:133] # safe slice; if dataset changes, still works

# Keep only numeric columns in this block
personality_block = [c for c in personality_block if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
print("Personality block size:", len(personality_block))

# Standardize and compute PCA(5) for trait-like scores
trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Emotional_Stability"]

if len(personality_block) >= 10:
    scaler_traits = StandardScaler()
    Z = scaler_traits.fit_transform(df[personality_block].values)
    pca = PCA(n_components=5, random_state=42)
    traits = pca.fit_transform(Z)

    for i, name in enumerate(trait_names):
        df[name] = traits[:, i]
else:
    # fallback: averages if block is missing
    for name in trait_names:
        df[name] = 0.0

# Consistency metrics
df["Likert_Mean"] = df[likert_cols].mean(axis=1)
df["Likert_Var"] = df[likert_cols].var(axis=1)
df["Likert_Consistency"] = 1.0 / (1.0 + df["Likert_Var"]) # higher = more consistent answers

# Composite index example
df["Introversion_Extroversion_Index"] = df["Extraversion"] # (already centered)

# BMI (Height is cm here)
if "Height" in df.columns and "Weight" in df.columns:
    df["BMI"] = df["Weight"] / (df["Height"] / 100.0) ** 2

# ------------------------------------------

# --- Page 5 ---
# One-hot encode low-cardinality categoricals
# ------------------------------------------
low_card_cols = []
for c in cat_cols + (["age_bin"] if "age_bin" in df.columns else []):
    if c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun <= 20:
            low_card_cols.append(c)

df_model = df.copy()
if low_card_cols:
    df_model = pd.get_dummies(df_model, columns=low_card_cols, drop_first=True)

# ------------------------------------------
# EDA
# ------------------------------------------
print("\n--- EDA ---")

# Target distribution example
TARGET = "Happiness in life"
if TARGET in df.columns:
    plt.figure()
    df[TARGET].hist(bins=10)
    plt.title("Target distribution: Happiness in life")
    plt.xlabel(TARGET)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png")
    plt.show(block=False)
    plt.close()

# Correlation heatmap (numeric)
num_for_corr = [c for c in df_model.columns if pd.api.types.is_numeric_dtype(df_model[c])]
corr = df_model[num_for_corr].corr()

# plot only a manageable subset (top 30 by variance)
var_series = df_model[num_for_corr].var().sort_values(ascending=False)
top_cols = var_series.head(30).index.tolist()

plt.figure(figsize=(10, 8))
plt.imshow(df_model[top_cols].corr(), aspect="auto")
plt.xticks(range(len(top_cols)), top_cols, rotation=90, fontsize=7)
plt.yticks(range(len(top_cols)), top_cols, fontsize=7)
plt.title("Correlation heatmap (Top 30 numeric features)")
plt.colorbar()
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png")
plt.show(block=False)
plt.close()

# Gender-based differences in traits (if Gender exists)
if "Gender" in df.columns:
    if "gender" in df.columns: # in case column got normalized

# --- Page 6 ---
        gender_col = "gender"
    else:
        gender_col = "Gender"

    # Try to compare mean traits by gender if it exists after normalization
    if gender_col in df.columns:
        g = df.groupby(gender_col)[trait_names].mean()
        print("\nMean trait scores by gender:")
        print(g)

# ------------------------------------------
# Clustering of participants based on trait scores
# ------------------------------------------
cluster_features = [c for c in trait_names + ["Likert_Mean", "Likert_Consistency"] if c in df_model.columns]
X_cluster = df_model[cluster_features].values

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_model["Personality_Cluster"] = kmeans.fit_predict(X_cluster)

print("\nCluster counts:")
print(df_model["Personality_Cluster"].value_counts())

# ------------------------------------------
# Regression models: predict Happiness in life
# ------------------------------------------
print("\n--- ML: Regression ---")

if TARGET not in df_model.columns:
    raise ValueError("Target column not found: " + TARGET)

# Choose input features: traits + demographics + lifestyle indicators
# drop all raw survey columns except a subset + encoded demographics
base_inputs = trait_names + ["Age", "Height", "Weight", "BMI", "Likert_Mean", "Likert_Consistency"]
base_inputs = [c for c in base_inputs if c in df_model.columns]

# Add one-hot demographics if present
demo_cols = [c for c in df_model.columns if any(c.startswith(prefix) for prefix in [
    "Gender_", "Education_", "Smoking_", "Alcohol_", "Village - town_", "House - block of flats_", "age_bin_"
])]
input_cols = base_inputs + demo_cols
input_cols = list(dict.fromkeys(input_cols)) # unique, keep order

X = df_model[input_cols].copy()

# --- Page 7 ---
y = df_model[TARGET].astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale for linear and KNN models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = []
models.append(("LinearRegression", LinearRegression(), True))
models.append(("KNNRegressor", KNeighborsRegressor(n_neighbors=15), True))
models.append(("RandomForest", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1), False))

models.append(("XGBoost", XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, random_state=42
), False))

results = []

for name, model, needs_scaled in models:
    if needs_scaled:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("MAE")
print("\nModel comparison (lower MAE is better):")
print(results_df)

plt.figure()
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Model comparison by MAE (Happiness in life)")
plt.ylabel("MAE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison_mae.png")
plt.show(block=False)
plt.close()

best_model_name = results_df.iloc[0]["Model"]
print("\nBest model:", best_model_name)

# ------------------------------------------
# ROC curve for best model (classification version)
# ------------------------------------------
print("\n--- ROC Curve (Classification) ---")

# Convert Happiness into High (>=4) vs Low (<4)
y_bin = (df_model[TARGET] >= 4).astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_bin, test_size=0.25, random_state=42, stratify=y_bin
)

clf_candidates = []
clf_candidates.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1), False))

clf_candidates.append(("XGBoostClassifier", XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss"
), False))

best_auc = -1
best_clf = None
best_clf_name = None

for cname, clf, needs_scaled in clf_candidates:
    clf.fit(X_train_c, y_train_c)
    prob = clf.predict_proba(X_test_c)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, prob)
    roc_auc = auc(fpr, tpr)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_clf = clf
        best_clf_name = cname
        best_curve = (fpr, tpr)

print("Best classifier for ROC:", best_clf_name, "AUC:", round(best_auc, 3))

fpr, tpr = best_curve
plt.figure()
plt.plot(fpr, tpr, label="AUC = %.3f" % best_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# --- Page 9 ---
plt.title("ROC Curve: High Happiness (>=4) vs Low")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_happiness.png")
plt.show(block=False)
plt.close()

# ------------------------------------------
# Practical prediction utilities
# ------------------------------------------
print("\n--- Practical Predictions ---")

# Personality trait prediction: we already computed trait-like scores.
print("Example trait columns:", trait_names)

# Cluster assignment for a new person:
def predict_cluster_for_new_person(new_row_dict):
    """
    new_row_dict: dict with trait-like features used for clustering.
    Required keys (recommended): Openness, Conscientiousness,
    Extraversion, Agreeableness,
    Emotional_Stability, Likert_Mean, Likert_Consistency

    Returns: predicted cluster id (0..k-1)
    """
    # Build a single-row dataframe using the same cluster_features
    cfeat = {k: new_row_dict.get(k, 0) for k in cluster_features}
    temp2 = pd.DataFrame([cfeat])[cluster_features]
    temp2 = scaler_cluster.transform(temp2)
    return int(kmeans.predict(temp2)[0])

# Predict happiness (regression) for a new person:
def predict_happiness(new_row_dict):
    """
    Uses the best regression model from the comparison table.
    Returns: predicted happiness score (approx).
    """
    temp = pd.DataFrame([new_row_dict])
    for c in input_cols:
        if c not in temp.columns:
            temp[c] = 0
    temp = temp[input_cols]
    # choose best model again
    name_to_model = {name: (model, scaled) for name, model, scaled in models}
    model, scaled = name_to_model[best_model_name]
    if scaled:
        temp_s = scaler.transform(temp)
        return float(model.predict(temp_s)[0])
    return float(model.predict(temp)[0])

# recommendation example:

# --- Page 10 ---
# Recommend top 5 "interest" columns (31..62) based on similarity to
# nearest neighbors in trait space.
interest_cols = cols[31:63]
interest_cols = [c for c in interest_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def recommend_interests(person_index, top_k=5):
    """
    For a given person row index, find nearest neighbors in trait
    space and recommend interests they rate highly on average.
    """
    if person_index < 0 or person_index >= len(df_model):
        return []
    traits_mat = df_model[trait_names].values
    # Euclidean distance
    x0 = traits_mat[person_index]
    d = np.sqrt(((traits_mat - x0) ** 2).sum(axis=1))
    nn_idx = np.argsort(d)[1:51] # top 50 neighbors (excluding self)
    if not interest_cols:
        return []
    mean_interest = df.loc[nn_idx, interest_cols].mean().sort_values(ascending=False)
    return list(mean_interest.head(top_k).index)

# Demo with first row
demo_row = df_model.iloc[0]
new_person = {c: float(demo_row[c]) for c in input_cols if pd.api.types.is_numeric_dtype(df_model[c])}
# Note: The line above was slightly adjusted from the image to fix a small logic gap in the image text
# Image text: new_person = {c: float(demo_row[c]) } if ... else 0.
# I will use the image version below for strict accuracy to the source:

new_person = {c: float(demo_row[c]) if pd.api.types.is_numeric_dtype(df_model[c]) else 0 for c in input_cols}

pred_h = predict_happiness(new_person)
pred_cluster = predict_cluster_for_new_person({**{k: float(demo_row[k]) for k in cluster_features}, **new_person})
recs = recommend_interests(0, top_k=5)

print("Predicted Happiness (row 0-like):", round(pred_h, 3))
print("Predicted Personality Cluster:", pred_cluster)
print("Recommended interest columns (top 5):", recs)

# Save structured results for web dashboard
import json
results_payload = {
    "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    "detected_likert_count": len(likert_cols),
    "personality_block_size": len(personality_block),
    "cluster_counts": {str(k): int(v) for k, v in df_model["Personality_Cluster"].value_counts().items()},
    "mean_traits_by_gender": g.round(4).to_dict(orient="index") if "g" in locals() and isinstance(g, pd.DataFrame) else {},
    "model_comparison": results_df.round(4).to_dict(orient="records"),
    "best_model": str(best_model_name),
    "best_classifier": str(best_clf_name),
    "best_auc": round(float(best_auc), 3),
    "demo_prediction": {
        "happiness": round(float(pred_h), 3),
        "cluster": int(pred_cluster),
        "recommended_interests": recs
    },
    "trait_names": trait_names,
    "input_cols": input_cols
}
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2)
print("Results successfully saved to:", results_path)
