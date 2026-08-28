import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Workforce Stability Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# NEW CUSTOM CSS (Burgundy & Black Theme)
# ============================================================
# تم تحديث الألوان هنا لتطابق الصورة تماماً
primary_burgundy = "#6B0103"  # لون البورغندي الأساسي
dark_bg = "#000000"  # خلفية داكنة جداً
card_bg = "#111111"  # خلفية البطاقات
text_light = "#FFFFFF"  # نص أبيض
text_muted = "#AAAAAA"  # نص رمادي خافت

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Font & Main Background */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {dark_bg};
        color: {text_light};
    }}
    .stApp {{
        background-color: {dark_bg};
    }}

    /* Sidebar Styling - Dark & High Contrast */
    section[data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid #222;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] div {{
        color: {text_light} !important;
    }}
    /* Fix dropdown visibility in sidebar */
    section[data-testid="stSidebar"] .st-ak {{ 
        color: {dark_bg} !important; 
    }}
    /* Sidebar Title Style */
    .sidebar-title {{
        font-weight: 800; font-size: 1.2rem; color: {primary_burgundy} !important; margin-bottom: 0.5rem;
    }}
    .sidebar-subtitle {{
        font-size: 0.85rem; color: {text_muted} !important; margin-bottom: 1rem;
    }}

    /* Header */
    .dashboard-header {{
        background: linear-gradient(135deg, {primary_burgundy} 0%, #330000 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(107, 1, 3, 0.3);
    }}
    .dashboard-header h1 {{ color: #FFFFFF; font-size: 32px; font-weight: 800; margin: 0; }}
    .dashboard-header p {{ color: #FFCCCC; margin-top: 8px; font-size: 15px; opacity: 0.9; }}

    /* KPI Cards - Glassmorphism style on dark */
    .kpi-card {{
        background: {card_bg};
        border-radius: 10px;
        padding: 25px;
        border: 1px solid #222;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: {primary_burgundy};
    }}
    .kpi-label {{ font-size: 13px; font-weight: 600; color: {text_muted}; text-transform: uppercase; letter-spacing: 1px; }}
    .kpi-value {{ font-size: 32px; font-weight: 800; color: #FFFFFF; margin-top: 5px; }}
    .kpi-description {{ font-size: 11px; color: #777777; margin-top: 4px; }}

    /* Section Titles with Burgundy underline */
    .section-title {{
        font-size: 20px; font-weight: 700; color: #FFFFFF;
        margin-top: 40px; margin-bottom: 20px;
        padding-bottom: 10px; border-bottom: 3px solid {primary_burgundy};
    }}

    /* Insight Cards - Burgundy accents */
    .insight-card {{
        background: {card_bg}; border-radius: 8px; padding: 18px;
        margin-bottom: 12px; border-left: 5px solid {primary_burgundy};
        box-shadow: 0 2px 5px rgba(0,0,0,0.2); color: {text_light}; font-size: 14px;
    }}

    /* Risk Cards - High Contrast on Dark */
    .risk-high {{ background: #330000; border: 2px solid #FF0000; color: #FFCCCC; padding: 25px; border-radius: 12px; text-align: center; box-shadow: 0 0 15px rgba(255,0,0,0.2); }}
    .risk-medium {{ background: #331A00; border: 2px solid #FFA500; color: #FFE0B3; padding: 25px; border-radius: 12px; text-align: center; }}
    .risk-low {{ background: #001A00; border: 2px solid #00FF00; color: #B3FFB3; padding: 25px; border-radius: 12px; text-align: center; }}
    .risk-number {{ font-size: 48px; font-weight: 800; margin: 10px 0; }}
    .risk-label {{ font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #FFFFFF; }}

    /* Dataframe Visibility Fix on Dark */
    .stDataFrame {{ background-color: {card_bg}; border: 1px solid #222 !important; border-radius: 8px !important; color: {text_light}; }}
    div[data-testid="stMetricValue"] {{ color: {text_light} !important; font-weight: 700 !important; }}
    div[data-testid="stMetricLabel"] {{ color: {text_muted} !important; font-weight: 600 !important; }}

    /* Buttons */
    .stButton>button {{
        background-color: {primary_burgundy} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: #880000 !important;
        box-shadow: 0 0 10px rgba(107, 1, 3, 0.5) !important;
    }}

    /* Footer */
    .footer {{ text-align: center; color: #555555; font-size: 12px; padding: 30px 0; border-top: 1px solid #222; margin-top: 50px; }}
</style>
""", unsafe_allow_html=True)

# Define a custom color palette for charts to match
chart_colors = [primary_burgundy, "#FF4444", "#FFAAAA", "#330000"]
plot_template = "plotly_dark"

# ============================================================
# FILE PATHS
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
        st.error(f"⚠️ ملف البيانات غير موجود: `{DATA_PATH}`")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    if "Age" in df.columns and "AgeGroup" not in df.columns:
        df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 65],
                                labels=["18-25", "26-35", "36-45", "46-55", "56-65"])
    return df


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists(): return None
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None


@st.cache_data
def load_model_columns():
    if not Path(COLUMNS_PATH).exists(): return None
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
    st.markdown('<div class="sidebar-title">◉ Workforce Stability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">By: Maram Ashraf & Sadeen Abdulrahman</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "📊 Executive Overview", "👥 Workforce Analytics",
        "⚠️ Attrition Drivers", "🤖 AI Risk Prediction", "🧠 Explainable AI"
    ])

    st.markdown("---")
    st.markdown("### 🔍 Filters")


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
    <h1>Workforce Stability Analyzer</h1>
    <p>Predictive HR Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)


# Helpers
def kpi(title, value, desc):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-description">{desc}<div class="kpi-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-description">{desc}</div>
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


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
        kpi("Workforce", f"{total_emp:,}", "Total employees")
    with c2:
        kpi("Attrition", f"{emp_left:,}", "Employees left")
    with c3:
        kpi("Churn Rate", f"{attrition_rate:.1f}%", "Overall percentage")
    with c4:
        kpi("Avg Income", f"${filtered_df['MonthlyIncome'].mean():,.0f}", "Monthly salary")

    section("Attrition Snapshot")
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart with Burgundy colors
        counts = filtered_df["Attrition"].value_counts().reset_index(name="Count")
        fig = px.pie(counts, names="Attrition", values="Count", hole=0.7,
                     color="Attrition", color_discrete_map={"No": "#444444", "Yes": primary_burgundy},
                     template=plot_template)
        fig.update_traces(textposition='outside', textinfo='percent+label', pull=[0, 0.1])
        fig.update_layout(height=400, showlegend=False, margin=dict(t=40, b=40, l=40, r=40))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Bar chart with Burgundy gradient
        dept_rate = calc_rate(filtered_df, "Department")
        fig = px.bar(dept_rate, x="Attrition Rate", y="Department", orientation="h",
                     text=dept_rate["Attrition Rate"].round(1).astype(str) + "%",
                     color="Attrition Rate", color_continuous_scale=[[0, "#330000"], [1, primary_burgundy]],
                     template=plot_template)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    section("Key Observations")
    insights = []
    ot_rate = calc_rate(filtered_df, "OverTime").set_index("OverTime")["Attrition Rate"]
    if not ot_rate.empty:
        insights.append(
            f"Attrition among **{ot_rate.idxmax()}** overtime workers is critically high at **{ot_rate.max():.1f}%**.")

    job_rate = calc_rate(filtered_df, "JobRole").set_index("JobRole")["Attrition Rate"]
    if not job_rate.empty:
        insights.append(f"**{job_rate.idxmax()}** is the role with the highest turnover (**{job_rate.max():.1f}%**).")

    for ins in insights:
        st.markdown(f'<div class="insight-card">⚠️ {ins}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2: WORKFORCE ANALYTICS
# ============================================================
elif page == "👥 Workforce Analytics":
    section("Key Metrics by Attrition State")


    # Custom styling for boxes to be Burgundy
    def make_box_plot(y_col, title):
        fig = px.box(filtered_df, x="Attrition", y=y_col, color="Attrition",
                     color_discrete_map={"No": "#888888", "Yes": primary_burgundy},
                     title=title, template=plot_template, points="outliers")
        fig.update_layout(height=380, showlegend=False)
        return fig


    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(make_box_plot("MonthlyIncome", "Income Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(make_box_plot("YearsAtCompany", "Tenure at Company"), use_container_width=True)
    with c3:
        st.plotly_chart(make_box_plot("DistanceFromHome", "Distance from Home"), use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        if "JobSatisfaction" in filtered_df.columns:
            sat_rate = calc_rate(filtered_df, "JobSatisfaction")
            sat_rate["JobSatisfaction"] = sat_rate["JobSatisfaction"].astype(str)
            fig = px.bar(sat_rate, x="JobSatisfaction", y="Attrition Rate", text="Attrition Rate",
                         title="Attrition Rate vs Job Satisfaction", template=plot_template,
                         color_discrete_sequence=[primary_burgundy])
            fig.update_traces(texttemplate='%{y:.1f}%', textposition="outside")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div style="margin-top:40px"></div>', unsafe_allow_html=True)
        kpi("Median Income", f"${filtered_df['MonthlyIncome'].median():,.0f}", "Filtered median")
        kpi("Avg Tenure", f"{filtered_df['YearsAtCompany'].mean():.1f} Yrs", "Filtered average")

# ============================================================
# PAGE 3: ATTRITION DRIVERS
# ============================================================
elif page == "⚠️ Attrition Drivers":
    section("Deep Dive into Drivers")
    drivers = ["OverTime", "JobRole", "Department", "AgeGroup", "BusinessTravel", "MaritalStatus", "EducationField"]
    valid_drivers = [d for d in drivers if d in filtered_df.columns]

    c1, c2 = st.columns([1, 4])
    with c1:
        sel_driver = st.radio("Analyze Factor:", valid_drivers)

    with c2:
        driver_data = calc_rate(filtered_df, sel_driver)
        driver_data[sel_driver] = driver_data[sel_driver].astype(str)

        fig = px.bar(driver_data, x="Attrition Rate", y=sel_driver, orientation="h",
                     text=driver_data["Attrition Rate"].round(1).astype(str) + "%",
                     color="Attrition Rate",
                     color_continuous_scale=[[0, "#222222"], [1, primary_burgundy]],
                     template=plot_template, title=f"Turnover Rate by {sel_driver}")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=550, coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4: AI RISK PREDICTION
# ============================================================
elif page == "🤖 AI Risk Prediction":
    section("Individual Attrition Risk Predictor")

    if model is None or model_columns is None:
        st.error("⚠️ AI Assets Mising. Ensure model and columns JSON exist.")
        st.stop()

    with st.expander("ℹ️ Input Employee Details", expanded=True):
        exclude_cols = ['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount', 'AgeGroup',
                        'EmployeeCount']
        prediction_cols = [col for col in model_columns if col not in exclude_cols and col in df.columns]

        with st.form("prediction_form"):
            input_values = {}
            cols = st.columns(3)

            for idx, col in enumerate(prediction_cols):
                with cols[idx % 3]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = st.number_input(col, value=float(df[col].median()))
                        input_values[col] = val
                    else:
                        opts = sorted(df[col].dropna().astype(str).unique().tolist())
                        val = st.selectbox(col, opts)
                        input_values[col] = val

            st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
            submit = st.form_submit_button("Run AI Risk Assessment")

    if submit:
        try:
            # Match columns exactly
            final_input = pd.DataFrame(columns=model_columns)
            temp_input = pd.DataFrame([input_values])
            for col in model_columns:
                if col in temp_input.columns:
                    final_input[col] = temp_input[col].values
                else:
                    final_input[col] = 0

            proba = model.predict_proba(final_input)[0][1]

            if proba >= 0.70:
                risk, css, icon = "HIGH RISK", "risk-high", "🔴"
            elif proba >= 0.40:
                risk, css, icon = "MEDIUM RISK", "risk-medium", "🟠"
            else:
                risk, css, icon = "LOW RISK", "risk-low", "🟢"

            st.markdown(f'<div class="section-title">Assessment Result</div>', unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"""
                <div class="{css}">
                    <div class="risk-label">{icon} {risk}</div>
                    <div class="risk-number">{proba * 100:.1f}%</div>
                    <div style="font-size:12px; opacity:0.8">Calculated Attrition Probability</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                # Gauge chart in Burgundy
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 20, 'color': text_light}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': text_muted},
                        'bar': {'color': primary_burgundy},
                        'bgcolor': "#222",
                        'borderwidth': 2,
                        'bordercolor': "#444",
                        'steps': [
                            {'range': [0, 40], 'color': '#003300'},
                            {'range': [40, 70], 'color': '#664400'},
                            {'range': [70, 100], 'color': '#440000'}],
                    }
                ))
                fig.update_layout(paper_bgcolor=dark_bg, plot_bgcolor=dark_bg, height=250, margin=dict(t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ============================================================
# PAGE 5: EXPLAINABLE AI
# ============================================================
elif page == "🧠 Explainable AI":
    section("AI Model Explainability (SHAP)")

    if not SHAP_FOLDER.exists():
        st.warning("⚠️ SHAP plots folder missing.")
    else:
        st.markdown(
            '<div class="insight-card">SHAP analysis helps understand which features most influence the AI model\'s predictions, providing transparency.</div>',
            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if Path(SHAP_FOLDER / "01_feature_importance.png").exists():
                st.image(str(SHAP_FOLDER / "01_feature_importance.png"), caption="Global Feature Importance",
                         use_container_width=True)
        with c2:
            if Path(SHAP_FOLDER / "02_shap_detailed.png").exists():
                st.image(str(SHAP_FOLDER / "02_shap_detailed.png"), caption="Feature Impact (Beeswarm)",
                         use_container_width=True)

        if Path(SHAP_FOLDER / "03_individual_prediction.png").exists():
            st.markdown('<div style="margin-top:30px"></div>', unsafe_allow_html=True)
            st.image(str(SHAP_FOLDER / "03_individual_prediction.png"), caption="Local Explanation (Single Case)",
                     use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
    <div class="footer">
        © 2026 Workforce Stability Analyzer | System Design by <b>Maram Ashraf</b> & <b>Sadeen Abdulrahman</b> 
        <br>
        <span style="opacity: 0.5; font-size: 10px;">Built with Streamlit & Plotly in Burgundy-Dark Theme</span>
    </div>
    """, unsafe_allow_html=True)