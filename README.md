# Social Media & Teen Mental Health Machine Learning Pipeline

An end-to-end Machine Learning production pipeline built to predict depression risk among teenagers based on their social media habits, sleep patterns, and daily activities. This project focuses on handling extreme class imbalance in mental health data using **SMOTE** and optimizing **Custom Decision Thresholds** to maximize prediction recall for high-risk cases.

---

## 📊 Project Overview & Objective
In medical and psychological domains, missing a high-risk individual (False Negative) is significantly more critical than a false alarm (False Positive). This project aims to build a highly sensitive screening pipeline that effectively captures teen depression risk.

### Core Architecture Features:
* **Robust Preprocessing:** Automated handling of categorical and numerical features using Scikit-Learn `ColumnTransformer`.
* **Imbalance Resolution:** Embedded `SMOTE` (Synthetic Minority Over-sampling Technique) inside an imbalanced-learn pipeline to prevent data leakage during cross-validation.
* **Threshold Optimization:** Lowered the decision threshold to `0.35` to significantly boost the **Recall** of the positive class (Depression).
* **Model Serialization:** Fully serialized pipeline using `joblib` for immediate production deployment.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & EDA:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Machine Learning Framework:** Scikit-Learn
* **Imbalance Framework:** Imbalanced-Learn (`imblearn`)
* **Advanced Boosting:** CatBoost

---

## 📈 Machine Learning Pipeline Structure

1. **Exploratory Data Analysis (EDA):** Analysed distributions, missing values, duplicates, and feature correlations. Identified a severe class imbalance in `depression_label`.
2. **Data Splitting:** Stratified split into 80% Training and 20% Testing sets.
3. **Column Transformation:** Applied `OneHotEncoder(handle_unknown='ignore')` for categorical variables while allowing numerical data to pass through safely.
4. **Model Exploration:** Evaluated 6 algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, GaussianNB, and CatBoost) via 5-fold Stratified Cross-Validation evaluating both `Accuracy` and `F1-Score`.
5. **Final Balancing & Tuning:** Combined the top-performing ensemble approach with SMOTE and fine-tuned decision thresholds.

---

## 🏆 Key Results & Performance

* **Baseline Models:** Tree-based ensembles (Random Forest, CatBoost) achieved top cross-validation metrics.
* **Threshold Adjustment impact:** By shifting the classification threshold from `0.5` to `0.35`, the model increased its sensitivity (**Recall**), ensuring that potential depression cases are not missed during screening while maintaining a robust overall accuracy.

---

## 🚀 How to Run the Project Locally

### 1. Clone the repository:
```bash
git clone [https://github.com//Yousef-salah-Ma/social-media-mental-health-ml.git](https://github.com//Yousef-salah-Ma/social-media-mental-health-ml.git)
cd social-media-mental-health-ml
