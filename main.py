import pandas as pd
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# first 5 rows of the data
print(df.head(5))
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


