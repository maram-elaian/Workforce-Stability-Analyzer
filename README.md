````markdown
# 🚀 Workforce Stability Analyzer

An interactive **HR Analytics and Machine Learning project** designed to analyze employee attrition, identify the main factors associated with employees leaving the company, and predict employee attrition risk.

The project combines **Exploratory Data Analysis (EDA), Data Visualization, Machine Learning, and Explainable AI (SHAP)** to provide useful insights into workforce stability.

---

## 🎯 Project Objective

Employee attrition is an important challenge for organizations because losing employees can lead to:

- Higher recruitment and training costs
- Loss of experienced employees
- Reduced productivity
- Increased workload on existing employees

The main goal of this project is to answer:

> **Why do employees leave the company, and can we predict which employees are at higher risk of leaving?**

---

## 📊 Dataset

This project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

The dataset contains employee information such as:

- Age
- Gender
- Department
- Job Role
- Monthly Income
- Job Satisfaction
- Overtime
- Business Travel
- Distance From Home
- Years at Company
- Total Working Years
- Job Level
- Work-Life Balance
- Performance Rating
- Relationship Satisfaction
- Attrition

### Target Variable

| Value | Meaning |
|---|---|
| `Yes` | Employee left the company |
| `No` | Employee stayed |

---

## 🧹 Data Preparation

The dataset was checked for:

- Missing values
- Duplicate rows
- Data types
- Descriptive statistics
- Unique values
- Unnecessary columns

The following columns were removed because they do not provide useful information for the analysis:

```text
EmployeeCount
Over18
StandardHours
EmployeeNumber
````

An additional feature called `AgeGroup` was created:

```text
18-25
26-35
36-45
46-55
56-65
```

---

## 🔎 Exploratory Data Analysis

EDA was performed to understand employee characteristics and identify patterns related to attrition.

### Main Visualizations

* Employee Attrition Distribution
* Attrition Rate by Overtime
* Attrition Rate by Job Role
* Attrition Rate by Age Group
* Monthly Income by Attrition
* Attrition Rate by Job Satisfaction
* Years at Company by Attrition
* Distance From Home by Attrition

Different visualization types were selected depending on the purpose of the analysis:

* **Count Plot** → categorical distributions
* **Bar Plot** → comparing attrition rates
* **Box Plot** → comparing numerical distributions between groups
* **Histogram** → understanding numerical distributions

---

## 📈 Key Findings

### ⏰ Overtime

Employees working overtime had a significantly higher attrition rate:

```text
OverTime = Yes → 30.53%
OverTime = No  → 10.44%
```

This makes overtime an important factor associated with employee attrition.

---

### 👔 Job Role

The highest attrition rates were found among:

```text
Sales Representative  → 39.76%
Laboratory Technician → 23.94%
Human Resources       → 23.08%
```

---

### 🎂 Age Group

The highest attrition rate was observed among employees aged 18–25:

```text
18-25 → 35.77%
26-35 → 19.14%
56-65 → 17.02%
46-55 → 11.50%
36-45 →  9.19%
```

---

### 😊 Job Satisfaction

Lower job satisfaction was associated with higher attrition:

```text
Satisfaction 1 → 22.84%
Satisfaction 2 → 16.43%
Satisfaction 3 → 16.52%
Satisfaction 4 → 11.33%
```

---

### 🕐 Years at Company

Employees who left had a lower average tenure:

```text
Stayed → 7.37 years
Left   → 5.13 years
```

---

### 💰 Monthly Income

Employees who stayed had a higher average monthly income:

```text
Stayed → 6832.74
Left   → 4787.09
```

---

### 🏠 Distance From Home

Employees who left had a higher average distance from home:

```text
Stayed → 8.92
Left   → 10.63
```

> These findings represent relationships and patterns in the dataset. They should not automatically be interpreted as causal relationships.

---

# 🤖 Machine Learning

A **Random Forest Classifier** was developed to predict employee attrition.

### Preprocessing

The Machine Learning pipeline includes:

* Train/Test Split
* StandardScaler for numerical features
* OneHotEncoder for categorical features
* Handling unknown categorical values
* Stratified train/test split
* Class balancing

The target variable was converted from:

```text
Yes → 1
No  → 0
```

### Random Forest Model

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
```

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

The confusion matrix is also visualized to understand correct and incorrect predictions.

---

# 🧠 Explainable AI — SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to interpret the Machine Learning model.

SHAP helps answer:

> **Why did the model predict that an employee is likely to leave?**

### SHAP Visualizations

#### 1. Feature Importance

Shows the most important features influencing employee attrition predictions.

#### 2. SHAP Beeswarm Plot

Shows how different feature values affect the model's attrition prediction.

#### 3. Waterfall Plot

Explains the prediction for an individual employee by showing which features increased or decreased the predicted risk.

The top 10 important features are also saved as:

```text
assets/shap_plots/top_10_features.csv
```

---

# 📊 Dashboard

The project includes an interactive **Streamlit dashboard** for exploring the analysis and model results.

The dashboard is designed to provide:

* Workforce overview
* Attrition statistics
* Attrition risk analysis
* Data visualizations
* Employee prediction
* Model explanations using SHAP

---

# 📁 Project Structure

```text
workforce-stability-analyzer/
│
├── data/
│   └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_comparison.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── pipeline.py
│   └── predict.py
│
├── models/
│   ├── attrition_model.pkl
│   └── model_columns.json
│
├── assets/
│   ├── charts/
│   └── shap_plots/
│       ├── 01_feature_importance.png
│       ├── 02_shap_detailed.png
│       ├── 03_individual_prediction.png
│       └── top_10_features.csv
│
├── docs/
│   └── recommendations.md
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest

### Explainable AI

* SHAP

### Model Saving

* Joblib

### Dashboard

* Streamlit

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/workforce-stability-analyzer.git
```

Navigate to the project directory:

```bash
cd workforce-stability-analyzer
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 👩‍💻 Team

### Sadeen Abdelrahman

**Data Analyst & Dashboard Development**

Responsibilities:

* Data preparation
* Exploratory Data Analysis
* Data visualization
* Business insights
* Streamlit dashboard development

### Maram Ashraf

**Machine Learning Engineer**

Responsibilities:

* Machine Learning preprocessing
* Model development
* Random Forest training
* Model evaluation
* SHAP analysis
* Prediction pipeline

---

# 🚀 Future Improvements

* Compare multiple Machine Learning models
* Hyperparameter tuning
* Improve prediction performance
* Add interactive employee risk prediction
* Add automated HR recommendations
* Improve dashboard visualizations
* Deploy the application online
* Add model monitoring

---

## ⚠️ Disclaimer

This project is developed for **educational and analytical purposes**.

Model predictions represent patterns learned from the dataset and should not be used as the sole basis for real-world employment decisions.

---

## ⭐ Project Goal

**Workforce Stability Analyzer transforms HR data into actionable insights to help organizations better understand employee attrition and support workforce retention.**

```
```
