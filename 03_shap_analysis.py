import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# بدء عملية تحليل SHAP
print("🔍 Starting SHAP Analysis for Model Interpretation...\n")

# 1. تحميل النموذج المدرب
model = joblib.load('models/attrition_model.pkl')
print("✅ Model loaded successfully\n")

# 2. تحميل البيانات النظيفة
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_clean.csv")
X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})

# إزالة الأعمدة غير المفيدة للتنبؤ
cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
X = X.drop(columns=[col for col in cols_to_drop if col in X.columns], errors='ignore')

print(f"📊 Number of features: {X.shape[1]}")
print(f"📊 Number of samples: {X.shape[0]}\n")

# 3. استخراج المعالج (Preprocessor) والنموذج (Classifier) من الـ Pipeline
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# تطبيق المعالجة المسبقة على البيانات
X_processed = preprocessor.transform(X)

# 4. تجهيز أسماء الميزات بعد الـ One-Hot Encoding
# الحصول على الأعمدة الرقمية والفئوية
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

# استخراج أسماء الأعمدة بعد الترميز
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()

# دمج جميع الأسماء
all_feature_names = numeric_features + cat_feature_names
all_feature_names = [str(name) for name in all_feature_names]

print(f"✅ Prepared {len(all_feature_names)} features for analysis\n")

# 5. إنشاء مجلد لحفظ الرسوم البيانية
os.makedirs('assets/shap_plots', exist_ok=True)

# 6. حساب قيم SHAP (قد يستغرق دقيقة أو دقيقتين)
print("⏳ Calculating SHAP Values (this may take 1-2 minutes)...")
explainer = shap.TreeExplainer(classifier)
shap_values_raw = explainer.shap_values(X_processed)

# استخراج قيم SHAP الخاصة بفئة التسرب (Yes = 1)
if isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[1]
else:
    if shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]
    else:
        shap_values = shap_values_raw

print("✅ SHAP Values calculated successfully!\n")

# ============================================
# الرسم الأول: Bar Plot لأهمية الميزات
# ============================================
print(" Creating Plot 1: Feature Importance (Bar Plot)...")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_processed, feature_names=all_feature_names, plot_type="bar", show=False)
plt.title("Top Factors Influencing Employee Attrition", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('assets/shap_plots/01_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()  # إغلاق النافذة لمنع تراكم الرسوم
print("   ✅ Saved: 01_feature_importance.png")

# ============================================
# الرسم الثاني: Beeswarm Plot (التأثير الإيجابي والسلبي)
# ============================================
print("📊 Creating Plot 2: Beeswarm Plot (Detailed Impact)...")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_processed, feature_names=all_feature_names, show=False)
plt.title("Feature Impact on Attrition Decision (Positive & Negative)", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('assets/shap_plots/02_shap_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 02_shap_detailed.png")

# ============================================
# استخراج أهم 10 عوامل وحفظها في ملف CSV
# ============================================
print("\n📊 Extracting Top 10 Most Important Features...")

# حساب متوسط القيمة المطلقة لقيم SHAP لكل ميزة
shap_importance = np.abs(shap_values).mean(axis=0).flatten()

# إنشاء DataFrame وترتيب الميزات حسب الأهمية
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': shap_importance
}).sort_values('Importance', ascending=False)

# أخذ أهم 10 ميزات
top_10_features = feature_importance_df.head(10)

# طباعة النتائج في الكونسول
print("\n" + "="*60)
print("🏆 Top 10 Factors Influencing Employee Attrition:")
print("="*60)
for i, row in top_10_features.iterrows():
    print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")
print("="*60)

# حفظ النتائج في ملف CSV
top_10_features.to_csv('assets/shap_plots/top_10_features.csv', index=False)
print("\n💾 Saved top features to: assets/shap_plots/top_10_features.csv")

# ============================================
# الرسم الثالث: Waterfall Plot (تحليل حالة فردية)
# ============================================
print("\n📊 Creating Plot 3: Individual Case Analysis (Waterfall Plot)...")

# اختيار أول موظف غادر فعلياً من البيانات
left_employee_idx = df[df['Attrition'] == 'Yes'].index[0]
sv_for_employee = shap_values[left_employee_idx]
base_value = explainer.expected_value[1]

# إنشاء كائن Explanation الخاص بـ SHAP
shap_exp = shap.Explanation(
    values=sv_for_employee,
    base_values=base_value,
    data=X_processed[left_employee_idx],
    feature_names=all_feature_names
)

# رسم مخطط الشلال (Waterfall)
plt.figure(figsize=(12, 8))
shap.plots.waterfall(shap_exp, max_display=10, show=False)
plt.title(f"Individual Analysis: Why did Employee #{left_employee_idx} leave?", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('assets/shap_plots/03_individual_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 03_individual_prediction.png")

# ============================================
# رسالة الانتهاء
# ============================================
print("\n" + "="*60)
print("🎉 All SHAP plots created successfully!")
print("📂 All plots and data saved in: assets/shap_plots/")
print("="*60)