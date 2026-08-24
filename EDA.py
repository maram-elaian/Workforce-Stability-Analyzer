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