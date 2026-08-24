import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_.csv")
# first 10 rows of the data
print(df.head(10))
print(f"Shape :{df.shape}")
print("INFO ABOUT DATA:\n",df.info)
print(df.describe())
# =====================================================
# NAMES OF THE COLUMNS
print(df.columns)
# ====================================================MISSING VALUES
print("Missing Values:")
print(df.isnull().sum())
# total  of Missing values in data
print("Total Missing Values:", df.isnull().sum().sum())
# ====================================================DUPLICATE ROWS
print("Duplicate Rows:")
print(df.duplicated().sum())
# ========================DELETE COLUMN
print(df.nunique().sort_values()) # BASED  on this we drop column
col=[ "EmployeeCount","Over18","StandardHours","EmployeeNumber"]
df.drop(columns=col, inplace=True)
print(df.shape)
# ================================data types
print("Data Types:")
print(df.dtypes)
# ==================
print(df["Attrition"].unique())
print(df["Gender"].unique())
print(df["Department"].unique())
print(df["JobRole"].unique())
print(df["OverTime"].unique())
print(df["BusinessTravel"].unique())
# ==================================outliers
sns.boxplot(x=df["MonthlyIncome"])
plt.show()
df.to_csv("WA_Fn-UseC_-HR-Employee-Attrition_clean.csv", index=False)

