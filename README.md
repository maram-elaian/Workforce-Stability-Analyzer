# 🚀 Workforce Stability Analyzer

An end-to-end **HR Analytics & Machine Learning** project that explores why employees leave, and predicts which employees are at risk of attrition — combining exploratory analysis, a Random Forest classifier, explainable AI (SHAP), and an interactive Streamlit dashboard.

> **Core question:** *Why do employees leave, and can we predict who's at risk before they do?*

---

## 🎯 Why It Matters

Employee attrition is costly — it drives up recruitment and training expenses, drains institutional knowledge, and overloads remaining staff. This project turns raw HR data into **actionable insight**, helping organizations spot risk factors early and act on them.

---

## 📊 Dataset

Built on the **IBM HR Analytics Employee Attrition & Performance** dataset, covering demographics, compensation, satisfaction, and tenure for each employee, with `Attrition` (Yes/No) as the target variable.

---

## 🔎 Key Findings

| Factor | Insight |
|---|---|
| ⏰ **Overtime** | Attrition nearly **3x higher** for employees working overtime (30.5% vs 10.4%) |
| 👔 **Job Role** | Sales Representatives have the highest attrition rate (**39.8%**) |
| 🎂 **Age** | Employees aged **18–25** are the most at-risk group (35.8%) |
| 😊 **Satisfaction** | Lowest satisfaction level correlates with the highest attrition (22.8%) |
| 💰 **Income** | Employees who left earned ~30% less on average |
| 🏠 **Commute** | Employees who left lived noticeably farther from work |

*These are patterns in the data, not proof of causation.*

---

## 🤖 Machine Learning & Explainability

A **Random Forest Classifier** (with class balancing and a Scikit-learn preprocessing pipeline) predicts attrition risk, evaluated with accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

**SHAP** is used to explain *why* the model makes each prediction — through feature importance, a beeswarm plot, and per-employee waterfall plots — so predictions stay transparent and trustworthy rather than a black box.

---

## 📊 Interactive Dashboard

A **Streamlit** dashboard brings it all together: workforce overview, attrition drivers, a live risk-prediction tool, and SHAP-based model explanations — all in one place.

---

## 🛠️ Tech Stack

`Python` · `Pandas` / `NumPy` · `Matplotlib` / `Seaborn` · `Scikit-learn` (Random Forest) · `SHAP` · `Joblib` · `Streamlit`

---

## ⚙️ Getting Started

```bash
git clone https://github.com/YOUR-USERNAME/workforce-stability-analyzer.git
cd workforce-stability-analyzer
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```text
workforce-stability-analyzer/
├── data/                 # Cleaned dataset
├── notebooks/            # EDA & model comparison
├── src/                  # Preprocessing, pipeline, prediction
├── models/                # Trained model + column schema
├── assets/shap_plots/     # SHAP visualizations
├── app.py                 # Streamlit dashboard
└── requirements.txt
```

---

## 👩‍💻 Team

| | |
|---|---|
| **Sadeen Abdelrahman** | Data Analyst & Dashboard Developer — data prep, EDA, visualization, Streamlit app |
| **Maram Ashraf** | Machine Learning Engineer — modeling, evaluation, SHAP analysis, prediction pipeline |

---

## 🚀 What's Next

Model comparison & hyperparameter tuning · automated HR recommendations · richer dashboard visuals · cloud deployment · model monitoring.

---

## ⚠️ Disclaimer

Built for **educational and analytical purposes**. Predictions reflect patterns in historical data and should not be the sole basis for real employment decisions.

---

**Workforce Stability Analyzer turns HR data into insight — helping organizations understand attrition and strengthen retention.**
