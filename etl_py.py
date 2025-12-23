import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Initial shape:", df.shape)

# =========================
# 2. BASIC INSPECTION
# =========================
print(df.info())
print(df.describe())

# =========================
# 3. DATA CLEANING
# =========================

# Convert TotalCharges to numeric (it comes as string)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values (new customers have no total charges)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop customerID (not useful for modeling)
df.drop(columns=["customerID"], inplace=True)

# =========================
# 4. TARGET VARIABLE
# =========================
# Encode churn: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 5. CLV FEATURE ENGINEERING
# =========================
# CLV = MonthlyCharges * tenure
df["CLV"] = df["MonthlyCharges"] * df["tenure"]

# High-value customer flag (top 25% CLV)
clv_threshold = df["CLV"].quantile(0.75)
df["HighValueCustomer"] = np.where(df["CLV"] >= clv_threshold, 1, 0)

# =========================
# 6. BINARY CATEGORICAL ENCODING
# =========================
binary_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

for col in binary_cols:
    df[col] = df[col].map({
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0
    })

# =========================
# 7. ONE-HOT ENCODING
# =========================
categorical_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

# =========================
# 8. FINAL CHECK
# =========================
print("Final shape:", df.shape)
print(df.head())

# =========================
# 9. SAVE CLEAN DATASET
# =========================
df.to_csv("telco_churn_clean.csv", index=False)

print("✅ Clean dataset saved as telco_churn_clean.csv")
