import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Create results directory
os.makedirs("results/models", exist_ok=True)

# Load preprocessed and balanced data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../data/balanced_sheet1.xlsx')
df = pd.read_excel(data_path, dtype={"Coach": str})

# Reverse any scaling done to Injury Location & Severity
# Reload original targets from unscaled original file
original_df = pd.read_excel(os.path.join(script_dir, '../data/sheet1.xlsx'))

df["Injury Severity"] = original_df["Injury Severity"]
df["Injury Location"] = original_df["Injury Location"]
df["Coach"] = original_df["Coach"]  # Ensure this matches for encoding too


# Encode Coach column
if "Coach" in df.columns:
    df["Coach"] = df["Coach"].fillna("Unknown")  # Handle NaNs
    if df["Coach"].dtype == "object":
        coach_encoder = LabelEncoder()
        df["Coach"] = coach_encoder.fit_transform(df["Coach"])
        os.makedirs("results/models", exist_ok=True)
        joblib.dump(coach_encoder, "results/models/coach_encoder.pkl")
        print("✅ Coach encoder saved.")
    else:
        print("⚠️ Coach column is not object type; skipping encoding.")
else:
    print("❌ 'Coach' column not found in data.")


# Separate features and targets
X = df.drop(columns=["Injury Severity", "Injury Location"])
y_severity = df["Injury Severity"]
y_location = df["Injury Location"]

# Encode all object/string columns in X (like Coach, Certification, etc.)
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    joblib.dump(le, f"results/models/{col.lower().replace(' ', '_')}_encoder.pkl")
    print(f"[ENCODED] {col}")

# Encode Injury Location and save encoder
location_encoder = LabelEncoder()
y_location_encoded = location_encoder.fit_transform(y_location)
joblib.dump(location_encoder, "results/models/location_encoder.pkl")
y_location = y_location_encoded

# Encode Injury Severity and save encoder
severity_encoder = LabelEncoder()
y_severity_encoded = severity_encoder.fit_transform(y_severity)
joblib.dump(severity_encoder, "results/models/severity_encoder.pkl")
y_severity = y_severity_encoded

# Ensure non-negative values
if y_location.min() < 0:
    shift = abs(y_location.min())
    y_location += shift
    print(f"Shifted Injury Location values up by {shift} to make all classes non-negative.")

# Split dataset
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_severity, test_size=0.2, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_location, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Neural Network": MLPClassifier(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train and evaluate for Injury Severity
print("\n===== Training for Injury Severity =====")
results_severity = []
for name, model in models.items():
    model.fit(X_train_s, y_train_s)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test_s, y_pred)
    f1 = f1_score(y_test_s, y_pred, average='weighted')
    report = classification_report(y_test_s, y_pred)
    print(f"\nModel: {name} (Severity)")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:\n", report)
    results_severity.append((name, acc, f1))
    joblib.dump(model, f"results/models/{name.replace(' ', '_').lower()}_severity_model.pkl")

# Train and evaluate for Injury Location
print("\n===== Training for Injury Location =====")
results_location = []
for name, model in models.items():
    model.fit(X_train_l, y_train_l)
    y_pred = model.predict(X_test_l)
    acc = accuracy_score(y_test_l, y_pred)
    f1 = f1_score(y_test_l, y_pred, average='weighted')
    report = classification_report(y_test_l, y_pred)
    print(f"\nModel: {name} (Location)")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:\n", report)
    results_location.append((name, acc, f1))
    joblib.dump(model, f"results/models/{name.replace(' ', '_').lower()}_location_model.pkl")

# Leaderboards
results_df_s = pd.DataFrame(results_severity, columns=["Model", "Accuracy", "F1 Score"])
results_df_l = pd.DataFrame(results_location, columns=["Model", "Accuracy", "F1 Score"])

results_df_s = results_df_s.sort_values(by="F1 Score", ascending=False)
results_df_l = results_df_l.sort_values(by="F1 Score", ascending=False)

print("\n--- Severity Model Leaderboard ---")
print(results_df_s)
print("\n--- Location Model Leaderboard ---")
print(results_df_l)

# Save leaderboards
results_df_s.to_csv("results/severity_model_leaderboard.csv", index=False)
results_df_l.to_csv("results/location_model_leaderboard.csv", index=False)
