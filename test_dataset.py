# test_dataset.py
# Complete dataset test: inspect, balance, preprocess, train, evaluate

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ====== UPDATE YOUR CSV PATH HERE ======
CSV_PATH = r"C:\Users\Aishwarya S P\Downloads\UNSW_2018_IoT_Botnet_Dataset_71.csv"

# ------------------------------------------------------------
# STEP 1: QUICK INSPECTION
# ------------------------------------------------------------
print("1) Quick inspect (first 5 rows & columns):")
df_sample = pd.read_csv(CSV_PATH, nrows=5)
print(df_sample.head())
print("\nColumns:", df_sample.columns.tolist())

# Count rows quickly
print("\nCounting rows (fast):")
with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
    row_count = sum(1 for _ in f) - 1
print("Approx rows:", row_count)

# ------------------------------------------------------------
# STEP 2: LOAD ENTIRE DATASET
# ------------------------------------------------------------
print("\nLoading full dataset...")
df = pd.read_csv(CSV_PATH, low_memory=False)

# ------------------------------------------------------------
# STEP 3: IDENTIFY LABEL COLUMN
# ------------------------------------------------------------
label_col = "DDoS"   # Your dataset's attack column

print(f"\nUsing label column: '{label_col}'")

print("\nFull label distribution:")
print(df[label_col].value_counts())

# ------------------------------------------------------------
# STEP 4: BALANCING DATASET (VERY IMPORTANT)
# ------------------------------------------------------------
print("\nBalancing dataset...")

from sklearn.utils import resample

df_majority = df[df[label_col] == "DDoS"]
df_minority = df[df[label_col] == "Normal"]

print("Majority (DDoS):", len(df_majority))
print("Minority (Normal):", len(df_minority))

# Oversample minority → get 5000 Normal rows
df_minority_up = resample(
    df_minority,
    replace=True,
    n_samples=5000,
    random_state=42
)

# Downsample majority → get 5000 DDoS rows
df_majority_down = resample(
    df_majority,
    replace=False,
    n_samples=5000,
    random_state=42
)

# Combine balanced dataset
df = pd.concat([df_minority_up, df_majority_down])
df = df.sample(frac=1, random_state=42)

print("\nBalanced distribution:")
print(df[label_col].value_counts())

# ------------------------------------------------------------
# STEP 5: CLEANING / PREPROCESSING
# ------------------------------------------------------------
# Replace objects & missing values
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].fillna("NA")
    else:
        df[c] = df[c].fillna(0)

# Encode categorical features
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != label_col]

from sklearn.preprocessing import LabelEncoder
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))

# Encode label
y = df[label_col].astype(str)
y_enc = LabelEncoder().fit_transform(y)

# Prepare feature matrix
X = df.drop(columns=[label_col]).values.astype(np.float32)

print("\nFeature matrix shape:", X.shape)
print("Labels shape:", y_enc.shape)

unique, counts = np.unique(y_enc, return_counts=True)
print("Label distribution (encoded):", dict(zip(unique, counts)))

# ------------------------------------------------------------
# STEP 6: TRAIN TEST SPLIT
# ------------------------------------------------------------
print("\nTraining a simple RandomForest model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------------------------------------
# STEP 7: EVALUATION
# ------------------------------------------------------------
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("\nTest Accuracy:", round(acc, 4))

print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))

# ------------------------------------------------------------
# STEP 8: SAVE SMALL SAMPLE
# ------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(CSV_PATH), "dataset_test_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sample_path = os.path.join(OUT_DIR, "sample_1000.csv")
df.sample(n=min(1000, len(df)), random_state=42).to_csv(sample_path, index=False)

print("\nSaved sample file:", sample_path)
print("Outputs saved in:", OUT_DIR)

print("\nDONE ✔ Dataset test + preprocessing + balancing complete.")
