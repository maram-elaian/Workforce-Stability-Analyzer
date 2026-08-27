import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from PIL import Image

# ==========================================
# 1. Color Palette - Navy + Teal Theme
# ==========================================
PRIMARY = "#1E3A5F"  # Navy Blue - اللون الأساسي
SECONDARY = "#2A9D8F"  # Teal - اللون الثانوي
BACKGROUND = "#F5F7FA"  # Light Gray - الخلفية
SUCCESS = "#2E8B57"  # Green - للنجاح/البقاء
DANGER = "#E76F51"  # Coral/Orange - للخطر/التسرب
TEXT = "#263238"  # Dark Gray - للنصوص
LIGHT_BLUE = "#E8F1F8"  # Light Blue - للخلفيات الفاتحة
WHITE = "#FFFFFF"


# ==========================================
# 2. CSS Styling
# ==========================================
def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Main Background */
    .stApp {{
        background: {BACKGROUND};
        font-family: 'Inter', sans-serif;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {PRIMARY};
        border-right: none;
    }}

    [data-testid="stSidebar"] .css-1d391kg {{
        background: transparent;
    }}

    /* Sidebar Text */
    .sidebar-title {{
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }}

    .sidebar-subtitle {{
        color: {SECONDARY};
        font-size: 0.9rem;
        font-weight: 500;
    }}

    /* Main Container */
    .main-container {{
        padding: 20px 40px;
        max-width: 1400px;
        margin: 0 auto;
    }}

    /* Headers */
    h1 {{
        color: {PRIMARY};
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 10px;
    }}

    h2 {{
        color: {PRIMARY};
        font-size: 1.8rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
    }}

    h3 {{
        color: {PRIMARY};
        font-weight: 600;
    }}

    /* KPI Cards */
    .kpi-card {{
        background: {WHITE};
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.08);
        border-left: 4px solid {SECONDARY};
        transition: all 0.3s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 4px 16px rgba(30, 58, 95, 0.15);
    }}

    .kpi-card.danger {{
        border-left-color: {DANGER};
    }}

    .kpi-card.success {{
        border-left-color: {SUCCESS};
    }}

    .kpi-label {{
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .kpi-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {PRIMARY};
        margin-bottom: 4px;
    }}

    .kpi-value.danger {{
        color: {DANGER};
    }}

    .kpi-value.success {{
        color: {SUCCESS};
    }}

    .kpi-subtext {{
        font-size: 0.8rem;
        color: #9CA3AF;
    }}

    /* Chart Containers */
    .chart-container {{
        background: {WHITE};
        border-radius: 12px;
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.08);
    }}

    .chart-title {{
        color: {PRIMARY};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 16px;
    }}

    /* Info Cards */
    .info-card {{
        background: {WHITE};
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.08);
        border-left: 4px solid {SECONDARY};
    }}

    .info-card.warning {{
        border-left-color: {DANGER};
        background: #FFF5F3;
    }}

    .info-card.success {{
        border-left-color: {SUCCESS};
        background: #F0F9F4;
    }}

    .info-card h4 {{
        color: {PRIMARY};
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .info-card p {{
        color: {TEXT};
        margin: 0;
        line-height: 1.6;
    }}

    /* Buttons */
    .stButton > button {{
        background: {PRIMARY};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        background: {SECONDARY};
        transform: translateY(-2px);
    }}

    /* Form Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {{
        background: {WHITE};
        border: 1px solid #D1D5DB;
        color: {TEXT};
        font-weight: 500;
        border-radius: 8px;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {{
        border-color: {SECONDARY};
    }}

    /* Divider */
    .divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #D1D5DB, transparent);
        margin: 30px 0;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 30px;
        color: #9CA3AF;
        font-size: 13px;
        margin-top: 50px;
        border-top: 1px solid #E5E7EB;
    }}

    /* Hide default alerts */
    .stAlert {{
        border-radius: 8px;
    }}

    </style>
    """, unsafe_allow_html=True)


load_css()

# ==========================================
# 3. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Workforce Stability Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 4. Load Data and Model
# ==========================================
@st.cache_data
def load_data():
    """تحميل البيانات النظيفة"""
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_clean.csv")


@st.cache_resource
def load_model():
    """تحميل النموذج المدرب"""
    return joblib.load('models/attrition_model.pkl')


# تحميل البيانات والنموذج
df = load_data()
model = load_model()

# ==========================================
# 5. Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 30px 20px;'>
        <div style='font-size: 3rem; margin-bottom: 10px;'>📊</div>
        <div class='sidebar-title'>Workforce<br>Stability</div>
        <div class='sidebar-subtitle'>Employee Attrition Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Navigation Menu
    page = st.radio(
        "Navigation:",
        [" Dashboard", "🔍 Risk Analysis", "🎯 Prediction", "💡 Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Filters
    st.markdown("<h4 style='color: white; margin-bottom: 15px;'> Filters</h4>", unsafe_allow_html=True)
    selected_dept = st.selectbox(
        "Department",
        ["All", "Sales", "Research & Development", "Human Resources"]
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color: rgba(255,255,255,0.7); font-size: 0.85rem; text-align: center;'>💡 AI-Powered Predictive Analytics</div>",
        unsafe_allow_html=True)

# ==========================================
# 6. Main Content
# ==========================================

if page == " Dashboard":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style='margin-bottom: 30px;'>
        <h1>📊 Workforce Analytics Dashboard</h1>
        <p style='color: #6B7280; font-size: 1.1rem;'>
            Comprehensive overview of employee attrition patterns and workforce stability metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total_employees = len(df)
    left_employees = (df['Attrition'] == 'Yes').sum()
    stayed_employees = (df['Attrition'] == 'No').sum()
    attrition_rate = (left_employees / total_employees) * 100
    avg_income = df['MonthlyIncome'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Employees</div>
            <div class='kpi-value'>{total_employees:,}</div>
            <div class='kpi-subtext'>👥 Active workforce</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='kpi-card danger'>
            <div class='kpi-label'>Employees Left</div>
            <div class='kpi-value danger'>{left_employees:,}</div>
            <div class='kpi-subtext'> Attrition cases</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='kpi-card success'>
            <div class='kpi-label'>Employees Stayed</div>
            <div class='kpi-value success'>{stayed_employees:,}</div>
            <div class='kpi-subtext'>✅ Retained talent</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Attrition Rate</div>
            <div class='kpi-value danger'>{attrition_rate:.1f}%</div>
            <div class='kpi-subtext'>📊 Turnover metric</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Charts Row 1
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Attrition Distribution</div>", unsafe_allow_html=True)

        attrition_counts = df['Attrition'].value_counts()
        fig1 = go.Figure(data=[
            go.Pie(
                labels=['Stayed', 'Left'],
                values=[attrition_counts.get('No', 0), attrition_counts.get('Yes', 0)],
                hole=0.5,
                marker_colors=[SUCCESS, DANGER],
                textinfo='percent+label',
                textposition='inside'
            )
        ])
        fig1.update_layout(showlegend=False, height=350, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Attrition by Department</div>", unsafe_allow_html=True)

        dept_attrition = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=dept_attrition.index,
            y=dept_attrition.get('No', pd.Series([0] * len(dept_attrition))),
            name='Stayed',
            marker_color=SUCCESS
        ))
        fig2.add_trace(go.Bar(
            x=dept_attrition.index,
            y=dept_attrition.get('Yes', pd.Series([0] * len(dept_attrition))),
            name='Left',
            marker_color=DANGER
        ))
        fig2.update_layout(barmode='group', height=350, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Charts Row 2
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Monthly Income by Attrition</div>", unsafe_allow_html=True)

        fig3 = px.box(df, x='Attrition', y='MonthlyIncome', color='Attrition',
                      color_discrete_map={'Yes': DANGER, 'No': SUCCESS})
        fig3.update_layout(showlegend=False, height=350, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart4:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Years at Company vs Attrition</div>", unsafe_allow_html=True)

        fig4 = px.box(df, x='Attrition', y='YearsAtCompany', color='Attrition',
                      color_discrete_map={'Yes': DANGER, 'No': SUCCESS})
        fig4.update_layout(showlegend=False, height=350, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif page == " Risk Analysis":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom: 30px;'>
        <h1>🔍 Risk Analysis</h1>
        <p style='color: #6B7280; font-size: 1.1rem;'>
            AI-powered analysis of key factors driving employee attrition using SHAP values
        </p>
    </div>
    """, unsafe_allow_html=True)

    # عرض رسوم SHAP المحفوظة
    shap_dir = 'assets/shap_plots'

    if os.path.exists(f'{shap_dir}/01_feature_importance.png'):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Top Factors Influencing Employee Attrition</div>", unsafe_allow_html=True)
        img1 = Image.open(f'{shap_dir}/01_feature_importance.png')
        st.image(img1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if os.path.exists(f'{shap_dir}/02_shap_detailed.png'):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Feature Impact on Attrition Decision (Positive & Negative)</div>",
                    unsafe_allow_html=True)
        img2 = Image.open(f'{shap_dir}/02_shap_detailed.png')
        st.image(img2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if os.path.exists(f'{shap_dir}/03_individual_prediction.png'):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Individual Case Analysis: Why Did This Employee Leave?</div>",
                    unsafe_allow_html=True)
        img3 = Image.open(f'{shap_dir}/03_individual_prediction.png')
        st.image(img3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # تحميل وعرض أهم العوامل
    if os.path.exists(f'{shap_dir}/top_10_features.csv'):
        top_features = pd.read_csv(f'{shap_dir}/top_10_features.csv')

        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Top 10 Risk Factors</div>", unsafe_allow_html=True)

        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=[SECONDARY, DANGER]
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🎯 Prediction":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom: 30px;'>
        <h1> Employee Attrition Prediction</h1>
        <p style='color: #6B7280; font-size: 1.1rem;'>
            Predict the likelihood of an employee leaving the company based on their profile
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            age = st.slider("Age", 18, 65, 30)
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
            job_level = st.slider("Job Level", 1, 5, 2)

        with col_f2:
            years_at_company = st.slider("Years at Company", 0, 40, 3)
            distance_from_home = st.slider("Distance from Home (km)", 1, 30, 10)
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

        with col_f3:
            overtime = st.selectbox("Over Time", ["No", "Yes"])
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                                 "Manufacturing Director", "Healthcare Representative",
                                                 "Manager", "Sales Representative", "Research Director", "HR"])

        submitted = st.form_submit_button("🔮 Predict Attrition Risk")

        if submitted:
            # إنشاء DataFrame للبيانات المدخلة
            input_data = pd.DataFrame({
                'Age': [age],
                'MonthlyIncome': [monthly_income],
                'JobLevel': [job_level],
                'YearsAtCompany': [years_at_company],
                'DistanceFromHome': [distance_from_home],
                'JobSatisfaction': [job_satisfaction],
                'OverTime': [overtime],
                'Department': [department],
                'JobRole': [job_role]
            })

            # إضافة أعمدة أخرى بقيم افتراضية (للتوافق مع النموذج)
            # في التطبيق الحقيقي، يجب إضافة جميع الأعمدة
            for col in df.columns:
                if col not in input_data.columns and col != 'Attrition':
                    if col in ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']:
                        continue
                    if df[col].dtype in ['int64', 'float64']:
                        input_data[col] = df[col].median()
                    else:
                        input_data[col] = df[col].mode()[0]

            # التنبؤ
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0][1]

                risk_score = int(probability * 100)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                # عرض النتيجة
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={'text': "Attrition Risk Score", 'font': {'size': 20, 'color': PRIMARY}},
                    number={'font': {'size': 40, 'color': DANGER if risk_score > 50 else SUCCESS}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': DANGER if risk_score > 50 else SUCCESS},
                        'steps': [
                            {'range': [0, 50], 'color': LIGHT_BLUE},
                            {'range': [50, 100], 'color': '#FFE5E0'}
                        ],
                        'threshold': {
                            'line': {'color': DANGER, 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=350)
                st.plotly_chart(fig_gauge, use_container_width=True)

                if risk_score > 50:
                    st.markdown(f"""
                    <div class='info-card warning'>
                        <h4>⚠️ High Risk: {risk_score}% Probability of Leaving</h4>
                        <p>This employee shows strong indicators of potential attrition. 
                        Immediate intervention is recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='info-card success'>
                        <h4>✅ Low Risk: {risk_score}% Probability of Leaving</h4>
                        <p>This employee appears stable. Continue regular engagement and development programs.</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "💡 Insights":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom: 30px;'>
        <h1>💡 Strategic Insights & Recommendations</h1>
        <p style='color: #6B7280; font-size: 1.1rem;'>
            Data-driven recommendations to improve employee retention and reduce attrition
        </p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = [
        {
            "icon": "⏰",
            "title": "Review Overtime Policy",
            "problem": "Employees working overtime show significantly higher attrition rates",
            "solution": "Redistribute workload, offer compensatory time off, or provide overtime bonuses",
            "type": "warning"
        },
        {
            "icon": "💰",
            "title": "Competitive Compensation",
            "problem": "Lower monthly income correlates with higher turnover",
            "solution": "Conduct market salary benchmarking and adjust compensation packages",
            "type": "warning"
        },
        {
            "icon": "📈",
            "title": "Career Development Programs",
            "problem": "Employees with fewer years at company are more likely to leave",
            "solution": "Create clear career paths and accelerate promotion opportunities for high performers",
            "type": "success"
        },
        {
            "icon": "😊",
            "title": "Improve Job Satisfaction",
            "problem": "Low job satisfaction is a strong predictor of attrition",
            "solution": "Implement regular pulse surveys and address concerns proactively",
            "type": "success"
        },
        {
            "icon": "🏠",
            "title": "Flexible Work Options",
            "problem": "Long commute distances increase attrition risk",
            "solution": "Offer remote work or hybrid work arrangements where possible",
            "type": "success"
        },
        {
            "icon": "🎯",
            "title": "Stock Options & Benefits",
            "problem": "Employees with lower stock option levels show higher turnover",
            "solution": "Review equity compensation and long-term incentive programs",
            "type": "success"
        }
    ]

    for rec in recommendations:
        st.markdown(f"""
        <div class='info-card {rec['type']}'>
            <h4>{rec['icon']} {rec['title']}</h4>
            <p><strong>🔴 Challenge:</strong> {rec['problem']}</p>
            <p><strong>💡 Recommendation:</strong> {rec['solution']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 7. Footer
# ==========================================
st.markdown("""
<div class='footer'>
    <p>📊 Workforce Stability Analyzer © 2024</p>
    <p style='font-size: 12px; margin-top: 10px;'>Powered by Machine Learning & Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)