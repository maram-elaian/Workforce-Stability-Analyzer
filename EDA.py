import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_clean.csv")
# ==============================Dataset Overview===========================
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
# ===================Attrition KPI==============
# Total Employees
Totalemployee=len(df)
print(f"the Total number of employee:{Totalemployee}")
# Employees Who Left
Employee_left=(df["Attrition"] == "Yes").sum()
print(f"number of employee who left :{Employee_left}")
# Employees Who Stayed
Employee_stay= (df["Attrition"] == "No").sum()
print(f"number of employee who stays :{Employee_stay}")
# Attrition Rate
attrition_rate=(Employee_left/Totalemployee)*100
print(f"Attrition Rate:{attrition_rate}")
kpi = pd.DataFrame({
    "Q": [
        "Total Employees",
        "Employees Left",
        "Employees Stayed",
        "Attrition Rate"
    ],
    "Value": [
        Totalemployee,
        Employee_left,
        Employee_stay,
        f"{attrition_rate:.2f}%"
    ]
})
# How many employee left the company compared with those who stayed?
plt.figure(figsize=(5,7))
sns.countplot(
    data=df,
    x="Attrition"
)
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()
"""Which age group is most represented?

Is the data normally distributed?

Are there too few or too many age groups?"""
print(df["Age"].describe())
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True
)
plt.title("Age Distribution of Employees")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()
# We want to know the distribution of employees by gender.
print(df["Gender"].value_counts())
plt.figure(figsize=(7, 5))
ax = sns.countplot(
    data=df,
    x="Gender"
)
plt.title("Employee Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()
# for the department
print(df["Department"].value_counts())
plt.figure(figsize=(8, 5))
dx = sns.countplot(
    data=df,
    x="Department"
)
plt.title("Employee Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()
