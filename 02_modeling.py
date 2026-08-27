import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import os
print("🚀 بدء عملية بناء النموذج...\n")

# 1. تحميل البيانات النظيفة
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_clean.csv")

# 2. تجهيز الـ Target Variable (تحويل Yes/No إلى 1/0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# 3. فصل الميزات (X) والهدف (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# إزالة الأعمدة التي لا فائدة منها للتنبؤ (إذا كانت لا تزال موجودة)
cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')

# 4. تحديد الأعمدة الرقمية والفئوية
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
print(f"✅ عدد الميزات الرقمية: {len(numeric_features)}")
print(f"✅ عدد الميزات الفئوية: {len(categorical_features)}\n")

# 5. بناء Pipeline للمعالجة المسبقة (أفضل ممارسة لتجنب تسرب البيانات)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),       # توحيد مقياس الأرقام
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # تحويل النصوص لأرقام
    ])

# 6. تقسيم البيانات (Train/Test Split)
# ملاحظة هامة: نستخدم stratify=y لضمان توازن نسبة التسرب في بيانات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ تم تقسيم البيانات:")
print(f"   - حجم بيانات التدريب: {X_train.shape}")
print(f"   - حجم بيانات الاختبار: {X_test.shape}\n")

# 7. بناء النموذج (Random Forest كبداية قوية)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

print("⏳ جاري تدريب النموذج...")
model_pipeline.fit(X_train, y_train)
print("✅ تم تدريب النموذج بنجاح!\n")

# 8. التقييم (Evaluation)
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("="*50)
print("📊 تقرير أداء النموذج (Classification Report):")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Left (1)']))

print(f"🎯 دقة النموذج (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"📈 مساحة تحت المنحنى (ROC-AUC): {roc_auc_score(y_test, y_pred_proba):.4f}")
print("="*50)

# 9. رسم مصفوفة الالتباس (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 10. حفظ النموذج لاستخدامه لاحقاً في الـ Dashboard
joblib.dump(model_pipeline, 'models/attrition_model.pkl')
print("\n💾 تم حفظ النموذج بنجاح في: models/attrition_model.pkl")
os.makedirs('models', exist_ok=True)

joblib.dump(model_pipeline, 'models/attrition_model.pkl')
print("\n💾 تم حفظ النموذج بنجاح في: models/attrition_model.pkl")