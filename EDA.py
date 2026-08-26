import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_clean.csv")
# ============================== Dataset Overview ============================
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
# ============================== Attrition KPI ===============================
# Total Employees
Totalemployee = len(df)
print(f"the Total number of employee: {Totalemployee}")
# Employees Who Left
Employee_left = (df["Attrition"] == "Yes").sum()
print(f"number of employee who left: {Employee_left}")
# Employees Who Stayed
Employee_stay = (df["Attrition"] == "No").sum()
print(f"number of employee who stays: {Employee_stay}")
# Attrition Rate
attrition_rate = (Employee_left / Totalemployee) * 100
print(f"Attrition Rate: {attrition_rate:.2f}%")
# KPI Table
kpi = pd.DataFrame({
    "Q": ["Total Employees",
        "Employees Left",
        "Employees Stayed",
        "Attrition Rate"],
    "Value":[Totalemployee,
        Employee_left,
        Employee_stay,
        f"{attrition_rate:.2f}%"]})
print(kpi)
# ============================================================
#ATTRITION DISTRIBUTION
# We want to know how many employees left and stayed.
plt.figure(figsize=(6, 5))
sns.countplot(
    data=df,
    x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.show()
# ============================================================
#ATTRITION BY OVERTIME
# We want to know if overtime is related to employee attrition.
overtime_attrition = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index") * 100
print(overtime_attrition)
plt.figure(figsize=(7, 5))
sns.barplot(
    data=overtime_attrition.reset_index(),
    x="OverTime",
    y="Yes")
plt.title("Attrition Rate by Overtime")
plt.xlabel("OverTime")
plt.ylabel("Attrition Rate (%)")
plt.tight_layout()
plt.show()
# ============================================================
#ATTRITION BY JOB ROLE
# We want to know which job roles have a higher attrition rate.
jobrole_attrition = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index") * 100
print(jobrole_attrition)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=jobrole_attrition.reset_index(),
    x="Yes",
    y="JobRole")
plt.title("Attrition Rate by Job Role")
plt.xlabel("Attrition Rate (%)")
plt.ylabel("Job Role")
plt.tight_layout()
plt.show()
# ============================================================
#ATTRITION BY AGE GROUP
# We want to know which age group has a higher attrition rate.
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 45, 55, 65],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
age_attrition = pd.crosstab(
    df["AgeGroup"],
    df["Attrition"],
    normalize="index") * 100
print(age_attrition)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=age_attrition.reset_index(),
    x="AgeGroup",
    y="Yes")
plt.title("Attrition Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Attrition Rate (%)")
plt.tight_layout()
plt.show()
# ============================================================
#MONTHLY INCOME BY ATTRITION
# We want to compare the income of employees who left and stayed.
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome")
plt.title("Monthly Income by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.show()
# ============================================================
# JOB SATISFACTION BY ATTRITION
# We want to know if job satisfaction is related to attrition.
satisfaction_attrition = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index") * 100
print(satisfaction_attrition)
plt.figure(figsize=(8, 5))
sns.barplot(
    data=satisfaction_attrition.reset_index(),
    x="JobSatisfaction",
    y="Yes")
plt.title("Attrition Rate by Job Satisfaction")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Attrition Rate (%)")
plt.tight_layout()
plt.show()
# ============================================================
#YEARS AT COMPANY BY ATTRITION
# We want to compare years at company between employees who left and stayed.
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany")
plt.title("Years at Company by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Years at Company")
plt.tight_layout()
plt.show()
# ============================================================
# DISTANCE FROM HOME BY ATTRITION
# We want to compare the distance from home between employees who left and stayed.
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Attrition",
    y="DistanceFromHome")
plt.title("Distance From Home by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Distance From Home")
plt.tight_layout()
plt.show()
# ============================================================
# BUSINESS INSIGHTS
# We want to find the main factors related to employee attrition.
# Overall Attrition Rate
print("Overall Attrition Rate:")
print(f"{attrition_rate:.2f}%")
# ============================================================
# Attrition Rate by Overtime
print("\nAttrition Rate by Overtime:")
overtime_rate = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index") * 100
print(overtime_rate["Yes"].sort_values(ascending=False))
# ============================================================
#Attrition Rate by Job Role
print("\nAttrition Rate by Job Role:")
jobrole_rate = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index") * 100
print(jobrole_rate["Yes"].sort_values(ascending=False))
# ============================================================
#Attrition Rate by Age Group
print("\nAttrition Rate by Age Group:")
age_rate = pd.crosstab(
    df["AgeGroup"],
    df["Attrition"],
    normalize="index") * 100
print(age_rate["Yes"].sort_values(ascending=False))
# ============================================================
#Attrition Rate by Job Satisfaction
print("\nAttrition Rate by Job Satisfaction:")
satisfaction_rate = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index") * 100
print(satisfaction_rate["Yes"].sort_values(ascending=False))
# ============================================================
#Attrition Rate by Department
print("\nAttrition Rate by Department:")
department_rate = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index") * 100
print(department_rate["Yes"].sort_values(ascending=False))
# ============================================================
#Average Years at Company by Attrition
print("\nAverage Years at Company by Attrition:")
print( df.groupby("Attrition")["YearsAtCompany"].mean())
# ============================================================
#Average Monthly Income by Attrition
print("\nAverage Monthly Income by Attrition:")
print(df.groupby("Attrition")["MonthlyIncome"].mean())
# ============================================================
# 9. Average Distance From Home by Attrition
print("\nAverage Distance From Home by Attrition:")
print(df.groupby("Attrition")["DistanceFromHome"].mean())
