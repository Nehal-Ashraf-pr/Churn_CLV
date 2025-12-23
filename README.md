# 📉 Churn Prediction & Customer Lifetime Value (CLV) Analytics

## 🔹 Project Overview
This project builds an **end-to-end churn analytics and decision system** for a subscription-based telecom business.  
It combines **machine learning, customer lifetime value (CLV), SQL analytics, and Power BI** to identify high-risk, high-value customers and estimate potential revenue uplift from retention actions.

The objective is to move beyond churn prediction and enable **business-driven decision making**.

---

## 🔹 Tech Stack
- **Python** – Pandas, NumPy, scikit-learn  
- **SQL Server (SSMS)** – Churn & CLV analytics  
- **Power BI** – Decision dashboard  
- **CSV (Telco Customer Churn Dataset)**  

---

## 🔹 Dataset
**Telco Customer Churn Dataset (Kaggle – BLASTCHAR)**

- ~7,000 customer records
- Subscription-based business model
- Explicit churn label

**Key fields:**
- `tenure`, `MonthlyCharges`, `TotalCharges`
- `Contract`, `PaymentMethod`, `InternetService`
- `Churn` (target variable)

---

## 🔹 Project Architecture

```text
Raw CSV
 ↓
Python (EDA + Feature Engineering + CLV)
 ↓
SQL Server (Churn & CLV Analytics)
 ↓
Python (Churn Prediction – Recall Optimized)
 ↓
Power BI (Churn × CLV Decision Dashboard)
```

## 🔹 Data Preparation & Feature Engineering (Python)

Steps:

- Cleaned billing fields and handled missing values

- Encoded categorical variables

- Created Customer Lifetime Value (CLV) feature:

- CLV = MonthlyCharges × tenure


- Flagged High-Value Customers (top 25% CLV)

- Prepared ML-ready dataset

📄 Output file:

- telco_churn_clean.csv

## 🔹 Churn Prediction Model

Model: Logistic Regression

- Objective: Maximize recall for high-value customers

- Techniques used:

- Class-balanced loss

- Threshold tuning

- Feature coefficient analysis

Model Performance

- Baseline churn recall: ~78%

- Recall after threshold tuning: 90%+

- Focused on minimizing missed high-CLV churners

## 🔹 SQL Analytics

Loaded the clean dataset into SQL Server and authored churn & CLV queries:

- Overall churn rate

- Churn rate by contract type

- CLV by churn status

- Revenue at risk (churned CLV)

- High-value customer churn

- Tenure-based churn cohorts

- Revenue uplift simulation (~8%)

## 🔹 Power BI Decision Dashboard

- Designed an executive decision dashboard combining churn risk and CLV.

Key KPIs

- Total Customers

- Churn Rate %

- Total CLV

- Revenue at Risk

- Projected Revenue Uplift (~8%)

Visuals

- Churn × CLV scatter plot (priority identification)

- Churn rate by contract type

- Revenue at risk by CLV segment

- Churn distribution

- Interactive slicers (Contract, Payment Method, CLV Segment)

```text
📸 Dashboard Preview
screenshots/
│── churn_clv_dashboard.png
│── churn_clv_scatter.png
```

## Churn × CLV Dashboard
<img width="1014" height="568" alt="image" src="https://github.com/user-attachments/assets/613ca355-05e8-45c8-b2e5-e12016e2a3d1" />

## Churn vs CLV Scatter
<img width="730" height="548" alt="image" src="https://github.com/user-attachments/assets/135979b3-9634-40c7-8969-78de968caa42" />

## 🔹 Business Insights

- Month-to-month contracts show the highest churn

- Majority of revenue at risk comes from high-CLV customers

- Recall optimization is critical for revenue protection

- Targeted retention can recover ~8% of at-risk revenue

## 🔹 Key Learnings

- Business metrics should guide ML optimization (recall over accuracy)

- CLV is essential for prioritizing churn actions

- Combining ML + BI enables decision-making, not just prediction

- SQL remains critical for validation and analysis

## 🔹 Future Enhancements

- Advanced models (XGBoost, LightGBM)

- Cost-sensitive learning

- Customer segmentation (RFM)

- Real-time churn monitoring

## 🔹 Author

- Nehal Ashraf
- Data Analyst | SQL | Python | Power BI | Machine Learning
