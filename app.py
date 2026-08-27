import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Workforce-Stability-Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS (تم إصلاح مشكلة الخطوط البيضاء والتباين)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background-color: #F8FAFC;
    }

    /* Sidebar Styling - Fixed Visibility */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #E2E8F0 !important;
    }
    /* Fix dropdown text color in sidebar */
    section[data-testid="stSidebar"] .st-ak { 
        color: #0F172A !important; 
    }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .dashboard-header h1 { color: #FFFFFF; font-size: 28px; font-weight: 800; margin: 0; }
    .dashboard-header p { color: #94A3B8; margin-top: 8px; font-size: 14px; }

    /* KPI Cards - Fixed Borders & Shadows */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        min-height: 110px;
    }
    .kpi-label { font-size: 12px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 26px; font-weight: 800; color: #0F172A; margin-top: 8px; }
    .kpi-description { font-size: 11px; color: #94A3B8; margin-top: 4px; }

    /* Section Titles */
    .section-title {
        font-size: 18px; font-weight: 700; color: #0F172A;
        margin-top: 35px; margin-bottom: 15px;
        padding-bottom: 8px; border-bottom: 2px solid #E2E8F0;
    }

    /* Insight Cards */
    .insight-card {
        background: #FFFFFF; border-radius: 10px; padding: 16px;
        margin-bottom: 10px; border-left: 4px solid #6366F1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04); color: #334155; font-size: 13px;
    }

    /* Risk Cards */
    .risk-high { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; padding: 24px; border-radius: 12px; text-align: center; }
    .risk-medium { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; padding: 24px; border-radius: 12px; text-align: center; }
    .risk-low { background: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; padding: 24px; border-radius: 12px; text-align: center; }
    .risk-number { font-size: 40px; font-weight: 800; margin: 10px 0; }
    .risk-label { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }

    /* Dataframe & Metrics Visibility Fix */
    .stDataFrame { border: 1px solid #E2E8F0 !important; border-radius: 8px !important; }
    div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { color: #64748B !important; font-weight: 600 !important; }

    /* Footer */
    .footer { text-align: center; color: #94A3B8; font-size: 11px; padding: 30px 0 10px; border-top: 1px solid #E2E8F0; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FILE PATHS (مطابقة تماماً لمخرجات أكوادك)
# ============================================================
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition_clean.csv"
MODEL_PATH = "models/attrition_model.pkl"
COLUMNS_PATH = "models/model_columns.json"
SHAP_FOLDER = Path("assets/shap_plots")


# ============================================================
# LOAD DATA & MODEL
# ============================================================
@st.cache_data
def load_data():
    if not Path(DATA_PATH).exists():
        st.error(f"⚠️ ملف البيانات غير موجود: `{DATA_PATH}`\nيرجى تشغيل كود تنظيف البيانات أولاً.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # إعادة إنشاء AgeGroup ليتطابق مع كود التحليل الخاص بك
    if "Age" in df.columns and "AgeGroup" not in df.columns:
        df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 65],
                                labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
    return df


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
        return None


@st.cache_data
def load_model_columns():
    if not Path(COLUMNS_PATH).exists():
        return None
    try:
        with open(COLUMNS_PATH, "r") as f:
            return json.load(f)
    except:
        return None


df = load_data()
model = load_model()
model_columns = load_model_columns()

# ============================================================
# SIDEBAR & FILTERS
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">◉Workforce-Stability-Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Employee Attrition Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-subtitle">By: Maram Ashraf & Sadeen Abdulrahman</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("---")

    page = st.radio("Navigation", [
        "📊 Executive Overview", "👥 Workforce Analytics",
        "⚠️ Attrition Drivers", "🤖 AI Risk Prediction", "🧠 Explainable AI"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Filters")


    # دالة مساعدة للفلاتر الديناميكية
    def filter_selectbox(col_name):
        if col_name in df.columns:
            opts = ["All"] + sorted(df[col_name].dropna().unique().tolist())
            return st.selectbox(col_name, opts)
        return "All"


    sel_dept = filter_selectbox("Department")
    sel_job = filter_selectbox("JobRole")
    sel_gender = filter_selectbox("Gender")
    sel_overtime = filter_selectbox("OverTime")

    age_range = None
    if "Age" in df.columns:
        min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))

# ============================================================
# APPLY FILTERS
# ============================================================
filtered_df = df.copy()
if sel_dept != "All": filtered_df = filtered_df[filtered_df["Department"] == sel_dept]
if sel_job != "All": filtered_df = filtered_df[filtered_df["JobRole"] == sel_job]
if sel_gender != "All": filtered_df = filtered_df[filtered_df["Gender"] == sel_gender]
if sel_overtime != "All": filtered_df = filtered_df[filtered_df["OverTime"] == sel_overtime]
if age_range: filtered_df = filtered_df[filtered_df["Age"].between(age_range[0], age_range[1])]

if filtered_df.empty:
    st.warning("لا توجد بيانات مطابقة للفلاتر المختارة.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="dashboard-header">
    <h1>Workforce-Stability-Analyzer</h1>
    <p>Data-driven workforce analytics, predictive risk assessment, and explainable AI.</p>
</div>
""", unsafe_allow_html=True)


# Helper for KPIs
def kpi(title, value, desc):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-description">{desc}</div>
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# دالة محسنة لحساب معدل التسرب (أسرع وأدق من apply)
def calc_rate(data, group_col):
    temp = data.copy()
    temp["Attrition_Num"] = (temp["Attrition"] == "Yes").astype(int)
    return temp.groupby(group_col, observed=True)["Attrition_Num"].mean().mul(100).reset_index(
        name="Attrition Rate").sort_values("Attrition Rate")


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================
if page == "📊 Executive Overview":
    total_emp = len(filtered_df)
    emp_left = (filtered_df["Attrition"] == "Yes").sum()
    attrition_rate = (emp_left / total_emp) * 100 if total_emp > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total Employees", f"{total_emp:,}", "Current filtered workforce")
    with c2:
        kpi("Employees Left", f"{emp_left:,}", "Total attrition count")
    with c3:
        kpi("Attrition Rate", f"{attrition_rate:.1f}%", "Overall attrition percentage")
    with c4:
        kpi("Avg. Monthly Income", f"${filtered_df['MonthlyIncome'].mean():,.0f}", "Average salary")

    section("Attrition Distribution")
    col1, col2 = st.columns(2)

    with col1:
        counts = filtered_df["Attrition"].value_counts().reset_index(name="Count")
        fig = px.pie(counts, names="Attrition", values="Count", hole=0.6,
                     color_discrete_sequence=["#10B981", "#EF4444"], template="plotly_white")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept_rate = calc_rate(filtered_df, "Department")
        fig = px.bar(dept_rate, x="Attrition Rate", y="Department", orientation="h",
                     text=dept_rate["Attrition Rate"].round(1).astype(str) + "%",
                     color="Attrition Rate", color_continuous_scale="Reds", template="plotly_white")
        fig.update_traces(textposition="inside")
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    section("Key Business Insights")
    insights = []
    ot_rate = calc_rate(filtered_df, "OverTime").set_index("OverTime")["Attrition Rate"]
    if not ot_rate.empty:
        insights.append(
            f"Employees with **{ot_rate.idxmax()}** overtime have the highest attrition rate (**{ot_rate.max():.1f}%**).")

    job_rate = calc_rate(filtered_df, "JobRole").set_index("JobRole")["Attrition Rate"]
    if not job_rate.empty:
        top_job = job_rate.idxmax()
        insights.append(f"**{top_job}** role shows the highest attrition rate at **{job_rate.max():.1f}%**.")

    if "AgeGroup" in filtered_df.columns:
        age_rate = calc_rate(filtered_df, "AgeGroup").set_index("AgeGroup")["Attrition Rate"]
        if not age_rate.empty:
            insights.append(f"The **{age_rate.idxmax()}** age group is the most vulnerable to attrition.")

    for ins in insights:
        st.markdown(f'<div class="insight-card">💡 {ins}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2: WORKFORCE ANALYTICS (مطابق لتحليلاتك)
# ============================================================
elif page == "👥 Workforce Analytics":
    section("Demographics & Compensation")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(filtered_df, x="Attrition", y="MonthlyIncome", color="Attrition",
                     title="Monthly Income by Attrition", template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filtered_df, x="Attrition", y="YearsAtCompany", color="Attrition",
                     title="Years at Company by Attrition", template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, x="Attrition", y="DistanceFromHome", color="Attrition",
                     title="Distance From Home by Attrition", template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "JobSatisfaction" in filtered_df.columns:
            sat_rate = calc_rate(filtered_df, "JobSatisfaction")
            sat_rate["JobSatisfaction"] = sat_rate["JobSatisfaction"].astype(str)
            fig = px.bar(sat_rate, x="JobSatisfaction", y="Attrition Rate", text="Attrition Rate",
                         title="Attrition Rate by Job Satisfaction Level", template="plotly_white",
                         color="Attrition Rate", color_continuous_scale="Reds")
            fig.update_traces(texttemplate='%{y:.1f}%', textposition="outside")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3: ATTRITION DRIVERS
# ============================================================
elif page == "⚠️ Attrition Drivers":
    section("Interactive Driver Analysis")
    drivers = ["OverTime", "JobRole", "Department", "AgeGroup", "BusinessTravel", "MaritalStatus"]
    valid_drivers = [d for d in drivers if d in filtered_df.columns]

    sel_driver = st.selectbox("Select Factor to Analyze", valid_drivers)

    driver_data = calc_rate(filtered_df, sel_driver)
    # تحويل الأعمدة لنص لضمان عرضها بشكل صحيح
    driver_data[sel_driver] = driver_data[sel_driver].astype(str)

    fig = px.bar(driver_data, x="Attrition Rate", y=sel_driver, orientation="h",
                 text=driver_data["Attrition Rate"].round(1).astype(str) + "%",
                 color="Attrition Rate", color_continuous_scale="Reds", template="plotly_white")
    fig.update_traces(textposition="inside")
    fig.update_layout(height=500, margin=dict(l=20, r=40, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4: AI RISK PREDICTION (مرتبط ديناميكياً بنموذجك)
# ============================================================
elif page == "🤖 AI Risk Prediction":
    section("AI Employee Risk Prediction")

    if model is None:
        st.error("⚠️ النموذج غير موجود. يرجى تشغيل كود تدريب النموذج أولاً لإنشاء `models/attrition_model.pkl`")
        st.stop()

    st.info("أدخل بيانات الموظف أدناه لتقدير احتمالية تركه للعمل بناءً على النموذج المدرب.")

    # تحديد الأعمدة المطلوبة ديناميكياً من ملف JSON الذي حفظه كودك
    if model_columns:
        # استبعاد الأعمدة التي حذفها كودك والأعمدة غير المطلوبة للتنبؤ
        exclude_cols = ['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount', 'AgeGroup']
        prediction_cols = [col for col in model_columns if col not in exclude_cols and col in df.columns]
    else:
        st.error("ملف model_columns.json غير موجود.")
        st.stop()

    with st.form("prediction_form"):
        input_values = {}
        cols = st.columns(3)

        for idx, col in enumerate(prediction_cols):
            with cols[idx % 3]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = st.number_input(col, min_value=float(df[col].min()),
                                          max_value=float(df[col].max()),
                                          value=float(df[col].median()))
                    input_values[col] = val
                else:
                    opts = sorted(df[col].dropna().astype(str).unique().tolist())
                    val = st.selectbox(col, opts)
                    input_values[col] = val

        submit = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submit:
        try:
            # إنشاء DataFrame وتطابقه تماماً مع أعمدة التدريب
            input_df = pd.DataFrame([input_values])

            # إعادة ترتيب الأعمدة لتطابق model_columns تماماً (مع ملء المفقود بـ 0 أو القيمة الافتراضية)
            # هذا يمنع أخطاء التنبؤ الناتجة عن اختلاف ترتيب أو وجود الأعمدة
            final_input = pd.DataFrame(columns=model_columns)
            for col in model_columns:
                if col in input_df.columns:
                    final_input[col] = input_df[col].values
                else:
                    final_input[col] = 0  # أو القيمة الافتراضية المناسبة

            # التنبؤ
            prediction = model.predict(final_input)[0]
            proba = model.predict_proba(final_input)[0]
            probability = proba[1] if len(proba) > 1 else proba[0]

            if probability >= 0.70:
                risk, css, icon = "HIGH RISK", "risk-high", "🔴"
            elif probability >= 0.40:
                risk, css, icon = "MEDIUM RISK", "risk-medium", "🟠"
            else:
                risk, css, icon = "LOW RISK", "risk-low", "🟢"

            st.markdown(f"""
            <div class="{css}">
                <div class="risk-label">{icon} {risk}</div>
                <div class="risk-number">{probability * 100:.1f}%</div>
                <div>Probability of Employee Attrition</div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Prediction", "Likely to Leave" if prediction == 1 else "Likely to Stay")
            with c2:
                st.metric("Stay Probability", f"{(1 - probability) * 100:.1f}%")

        except Exception as e:
            st.error(f"فشل التنبؤ. تأكد من تطابق البيانات: {e}")

# ============================================================
# PAGE 5: EXPLAINABLE AI (مربوط بمخرجات كود SHAP الخاص بك)
# ============================================================
elif page == "🧠 Explainable AI":
    section("Explainable AI (SHAP Analysis)")
    st.markdown("فهم العوامل المؤثرة في قرارات النموذج باستخدام تحليل SHAP.")

    if not SHAP_FOLDER.exists():
        st.warning("⚠️ مجلد `assets/shap_plots` غير موجود. يرجى تشغيل كود تحليل SHAP أولاً.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if Path(SHAP_FOLDER / "01_feature_importance.png").exists():
                st.image(str(SHAP_FOLDER / "01_feature_importance.png"), caption="Top Factors Influencing Attrition",
                         use_container_width=True)
        with col2:
            if Path(SHAP_FOLDER / "02_shap_detailed.png").exists():
                st.image(str(SHAP_FOLDER / "02_shap_detailed.png"), caption="Feature Impact (Beeswarm Plot)",
                         use_container_width=True)

        if Path(SHAP_FOLDER / "03_individual_prediction.png").exists():
            section("Individual Case Analysis")
            st.image(str(SHAP_FOLDER / "03_individual_prediction.png"), caption="Why did this specific employee leave?",
                     use_container_width=True)

        if Path(SHAP_FOLDER / "top_10_features.csv").exists():
            section("Top 10 Attrition Factors (Data)")
            top_df = pd.read_csv(SHAP_FOLDER / "top_10_features.csv")
            st.dataframe(top_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown('<div class="footer">HR Attrition Intelligence System • Built with Streamlit & Scikit-Learn</div>',
            unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer">
        © 2026 All rights reserved | Designed and developed by: <b>Maram Ashraf</b> & <b>Sadeen Abdulrahman</b> 
        <br>
        <span style="opacity: 0.7; font-size: 10px;">HR Attrition Intelligence System • Built with Streamlit, Scikit-Learn & SHAP</span>
    </div>
    """,
    unsafe_allow_html=True
)
