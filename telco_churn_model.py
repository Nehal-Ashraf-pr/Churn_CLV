import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score

df = pd.read_csv("telco_churn_clean.csv")
print(df.shape)


X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



high_value_mask = df.loc[y_test.index, "HighValueCustomer"] == 1

high_value_recall = recall_score(
    y_test[high_value_mask],
    y_pred[high_value_mask]
)

print("High-Value Customer Recall:", round(high_value_recall, 3))


y_prob = model.predict_proba(X_test_scaled)[:, 1]

custom_threshold = 0.35
y_pred_recall = (y_prob >= custom_threshold).astype(int)

print("Recall after threshold tuning:",
      recall_score(y_test, y_pred_recall))


feature_importance = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

print(feature_importance.head(10))
