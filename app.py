import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# ==========================================
# 1. CSS متقدم للتصميم الاحترافي
# ==========================================
def load_advanced_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 3px solid #9F7AEA;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }

    /* Navigation Buttons */
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid rgba(255,255,255,0.1);
    }

    .nav-button:hover {
        transform: translateX(10px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.5);
    }

    .nav-button.active {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: 2px solid white;
    }

    /* Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 2px solid rgba(255, 255, 255, 0.5);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #4A5568;
        font-weight: 600;
        margin-top: 5px;
    }

    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 30px;
        margin: 20px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    /* Headers */
    h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }

    h2 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 20px 0;
    }

    h3 {
        color: #764ba2;
        font-weight: 700;
    }

    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }

    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .info-card.warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .info-card.success {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 10px;
        border-radius: 10px;
    }

    /* Form Inputs */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #9F7AEA;
        color: #2D3748;
        font-weight: 600;
        border-radius: 10px;
    }

    /* Sidebar Text */
    .sidebar-text {
        color: #E0E0E0;
        font-size: 0.9rem;
    }

    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        margin: 20px 0;
    }

    /* Icon Styling */
    .icon {
        font-size: 2.5rem;
        margin-right: 10px;
    }

    /* Grid Layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        margin-top: 50px;
        border-top: 2px solid rgba(255,255,255,0.2);
    }

    /* Hide default Streamlit elements */
    .stAlert {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)


load_advanced_css()

# ==========================================
# 2. إعدادات الصفحة
# ==========================================
st.set_page_config(
    page_title="Workforce Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 3. بيانات وهمية
# ==========================================
@st.cache_data
def load_dummy_data():
    np.random.seed(42)
    n = 1470
    df = pd.DataFrame({
        'Age': np.random.randint(18, 65, n),
        'MonthlyIncome': np.random.randint(2000, 15000, n),
        'YearsAtCompany': np.random.randint(0, 40, n),
        'DistanceFromHome': np.random.randint(1, 30, n),
        'JobSatisfaction': np.random.randint(1, 5, n),
        'OverTime': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n),
        'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84])
    })
    return df


df = load_dummy_data()

# ==========================================
# 4. الشريط الجانبي - Navigation
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <div style='font-size: 3rem;'>💜</div>
        <h2 style='color: white; margin: 10px 0;'>Workforce<br>Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Navigation Menu
    page = st.radio(
        "اختر القسم:",
        ["📊 Dashboard", "🔍 Risk Analysis", " Prediction", "💡 Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Filters
    st.markdown("<h3 style='color: white;'>🔍 الفلاتر</h3>", unsafe_allow_html=True)
    selected_dept = st.selectbox(
        "القسم",
        ["All", "Sales", "Research & Development", "Human Resources"]
    )

    st.markdown(
        "<div class='sidebar-text' style='margin-top: 30px; text-align: center;'>💡 نظام ذكي للتنبؤ بتسرب الموظفين</div>",
        unsafe_allow_html=True)

# ==========================================
# 5. المحتوى الرئيسي حسب الصفحة المختارة
# ==========================================

if page == "📊 Dashboard":
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 40px;'>
        <h1>📊 Workforce Analytics Dashboard</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem;'>
            Advanced Predictive Analytics for Employee Retention
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total_employees = len(df)
    left_employees = (df['Attrition'] == 'Yes').sum()
    attrition_rate = (left_employees / total_employees) * 100
    avg_income = df['MonthlyIncome'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_employees:,}</div>
            <div class='metric-label'>👥 إجمالي الموظفين</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); -webkit-background-clip: text;'>
                {left_employees:,}
            </div>
            <div class='metric-label'>📉 غادروا الشركة</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); -webkit-background-clip: text;'>
                {attrition_rate:.1f}%
            </div>
            <div class='metric-label'> معدل التسرب</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); -webkit-background-clip: text;'>
                ${avg_income:,.0f}
            </div>
            <div class='metric-label'>💰 متوسط الراتب</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Charts Row 1
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig1 = px.pie(df, names='Attrition', title='<b>توزيع التسرب</b>',
                      color='Attrition',
                      color_discrete_map={'Yes': '#f5576c', 'No': '#667eea'},
                      hole=0.4)
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig2 = px.histogram(df, x='Department', color='Attrition',
                            title='<b>التسرب حسب القسم</b>',
                            color_discrete_map={'Yes': '#f5576c', 'No': '#667eea'},
                            barmode='group')
        fig2.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Charts Row 2
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig3 = px.box(df, x='OverTime', y='MonthlyIncome', color='Attrition',
                      title='<b>العمل الإضافي والراتب</b>',
                      color_discrete_map={'Yes': '#f5576c', 'No': '#667eea'})
        fig3.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart4:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig4 = px.scatter(df, x='YearsAtCompany', y='MonthlyIncome',
                          color='Attrition',
                          title='<b>سنوات الخبرة والراتب</b>',
                          color_discrete_map={'Yes': '#f5576c', 'No': '#667eea'})
        fig4.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔍 Risk Analysis":
    st.markdown("""
    <div style='text-align: center; padding: 40px;'>
        <h1>🔍 Risk Analysis</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem;'>
            تحليل عوامل الخطر المؤثرة في تسرب الموظفين
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Risk Factors
    risk_factors = pd.DataFrame({
        'العامل': ['العمل الإضافي', 'قلة سنوات الخبرة', 'بعد المسافة', 'انخفاض الرضا', 'الراتب المنخفض'],
        'نسبة التأثير': [45, 28, 18, 15, 12]
    })

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig_risk = px.bar(risk_factors, x='نسبة التأثير', y='العامل', orientation='h',
                      color='نسبة التأثير',
                      color_continuous_scale=px.colors.sequential.RdBu)
    fig_risk.update_layout(yaxis={'categoryorder': 'total ascending'},
                           height=500,
                           showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Insights Cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='info-card warning'>
            <h3>⚠️ تحذير: العمل الإضافي</h3>
            <p>الموظفون الذين يعملون OverTime لديهم معدل تسرب أعلى بـ 3 مرات</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card success'>
            <h3>💡 فرصة: تحسين الرضا</h3>
            <p>تحسين الرضا الوظيفي بنسبة 20% يقلل التسرب بنسبة 15%</p>
        </div>
        """, unsafe_allow_html=True)

elif page == " Prediction":
    st.markdown("""
    <div style='text-align: center; padding: 40px;'>
        <h1> Employee Prediction</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem;'>
            تنبؤ باحتمالية تسرب موظف معين
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            age = st.slider("🎂 العمر", 18, 65, 30)
            income = st.number_input("💰 الراتب الشهري", 1000, 20000, 5000)

        with col_f2:
            years = st.slider(" سنوات العمل", 0, 40, 3)
            distance = st.slider(" المسافة", 1, 30, 10)

        with col_f3:
            satisfaction = st.select_slider("😊 الرضا", [1, 2, 3, 4], 3)
            overtime = st.selectbox("⏰ العمل الإضافي", ["No", "Yes"])

        submitted = st.form_submit_button("🔮 احسب نسبة الخطر")

        if submitted:
            risk_score = 10
            if overtime == "Yes": risk_score += 40
            if years < 2: risk_score += 20
            if distance > 15: risk_score += 15
            if satisfaction <= 2: risk_score += 15
            risk_score = min(risk_score, 95)

            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "نسبة خطر التسرب", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#f5576c" if risk_score > 50 else "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(102, 126, 234, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(245, 87, 108, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if risk_score > 50:
                st.error(f"⚠️ خطر مرتفع: {risk_score}%")
            else:
                st.success(f"✅ خطر منخفض: {risk_score}%")

elif page == " Insights":
    st.markdown("""
    <div style='text-align: center; padding: 40px;'>
        <h1>💡 Strategic Insights</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem;'>
            توصيات استراتيجية للاحتفاظ بالموظفين
        </p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = [
        {"icon": "", "title": "مراجعة العمل الإضافي", "desc": "تقليل ساعات العمل الإضافي أو تعويضها"},
        {"icon": "", "title": "تسريع الترقية", "desc": "برامج تطوير للموظفين الجدد"},
        {"icon": "🏠", "title": "العمل المرن", "desc": "خيار العمل عن بعد أو الهجين"},
        {"icon": "😊", "title": "تحسين الرضا", "desc": "استبيانات دورية ومعالجة الشكاوى"}
    ]

    for i, rec in enumerate(recommendations):
        st.markdown(f"""
        <div class='info-card'>
            <h3>{rec['icon']} {rec['title']}</h3>
            <p>{rec['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. Footer
# ==========================================
st.markdown("""
<div class='footer'>
    <p>💜 Workforce Stability Analyzer © 2024</p>
    <p style='font-size: 12px; margin-top: 10px;'>Powered by Advanced AI Analytics</p>
</div>
""", unsafe_allow_html=True)