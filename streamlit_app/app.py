"""
Customer Churn Prediction System
Modern Professional Business Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Modern page configuration
st.set_page_config(
    page_title="Churn Analytics Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with professional design system
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Main content area */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-right: 3px solid #667eea;
    }
    
    /* Navigation styling */
    .nav-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Form styling */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Prediction result cards */
    .prediction-high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }
    
    .prediction-medium-risk {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(255, 167, 38, 0.3);
    }
    
    .prediction-low-risk {
        background: linear-gradient(135deg, #26c6da 0%, #00acc1 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(38, 198, 218, 0.3);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .status-info {
        background: linear-gradient(135deg, #2196f3 0%, #1565c0 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* About page styling */
    .about-section {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .capability-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .capability-card:hover {
        transform: translateX(5px);
    }
    
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    
    .developer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data with error handling"""
    try:
        df = pd.read_csv('data/processed/cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.error("🔴 Data not found. Please ensure the data processing pipeline has been completed.")
        return None

@st.cache_resource
def load_models_and_encoders():
    """Load models and encoders with comprehensive error handling"""
    try:
        data_summary = joblib.load('models/data_summary.pkl')
        encoders = joblib.load('models/encoders.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        models = {}
        model_files = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in model_files:
            try:
                models[model_name] = joblib.load(f'models/{model_name}.pkl')
            except FileNotFoundError:
                continue
        
        return data_summary, encoders, scaler, feature_names, models
    except FileNotFoundError:
        st.error("🔴 Model files not found. Please complete the model training pipeline.")
        return None, None, None, None, None

def main():
    """Main application with modern design"""
    
    # Hero header
    st.markdown('''
    <div class="hero-header">
        <h1>🎯 Customer Churn Analytics Platform</h1>
        <p>Enterprise-Grade Predictive Intelligence for Strategic Customer Retention</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    data_summary, encoders, scaler, feature_names, models = load_models_and_encoders()
    
    if df is None or data_summary is None:
        st.stop()
    
    # Modern sidebar navigation
    with st.sidebar:
        st.markdown("## 🎯 Navigation Center")
        st.markdown("---")
        
        page = st.radio(
            "",
            ["🏠 Executive Dashboard", "🔮 Churn Prediction", "📊 Analytics", "🤖 Model Performance", "ℹ️ About Platform"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        st.markdown(f'''
        <div class="status-card">
            <strong>✅ System Active</strong><br>
            <span style="color: #059669;">{data_summary['total_customers']:,} customers loaded</span>
        </div>
        ''', unsafe_allow_html=True)
        
        if models:
            st.markdown(f'''
            <div class="status-card" style="border-left-color: #3b82f6;">
                <strong>🤖 ML Models Ready</strong><br>
                <span style="color: #1d4ed8;">{len(models)} algorithms active</span>
            </div>
            ''', unsafe_allow_html=True)
    
    # Route to pages
    if page == "🏠 Executive Dashboard":
        show_executive_dashboard(df, data_summary)
    elif page == "🔮 Churn Prediction":
        show_prediction_interface(encoders, scaler, feature_names, models)
    elif page == "📊 Analytics":
        show_analytics_dashboard(df)
    elif page == "🤖 Model Performance":
        show_model_comparison(models)
    elif page == "ℹ️ About Platform":
        show_about_system()

def show_executive_dashboard(df, data_summary):
    """Modern executive dashboard with enhanced styling"""
    
    st.markdown('<div class="section-header"><h2>📊 Executive Intelligence Center</h2></div>', unsafe_allow_html=True)
    
    # Enhanced KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card-modern">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👥</div>
            <h2 style="color: #1e293b; margin: 0.5rem 0; font-size: 2.2rem; font-weight: 700;">{data_summary['total_customers']:,}</h2>
            <p style="color: #64748b; margin: 0; font-weight: 500;">Active Customer Base</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        churn_count = data_summary['churned_customers']
        st.markdown(f'''
        <div class="metric-card-modern" style="border-top: 5px solid #ef4444;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📉</div>
            <h2 style="color: #ef4444; margin: 0.5rem 0; font-size: 2.2rem; font-weight: 700;">{churn_count:,}</h2>
            <p style="color: #64748b; margin: 0; font-weight: 500;">Churned Customers</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        churn_rate = data_summary['churn_rate']
        
        if churn_rate > 0.25:
            color = "#ef4444"
            icon = "🚨"
        elif churn_rate > 0.15:
            color = "#f59e0b"
            icon = "⚠️"
        else:
            color = "#10b981"
            icon = "✅"
        
        st.markdown(f'''
        <div class="metric-card-modern" style="border-top: 5px solid {color};">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <h2 style="color: {color}; margin: 0.5rem 0; font-size: 2.2rem; font-weight: 700;">{churn_rate:.1%}</h2>
            <p style="color: #64748b; margin: 0; font-weight: 500;">Churn Rate</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        avg_revenue = data_summary['avg_monthly_charges']
        st.markdown(f'''
        <div class="metric-card-modern" style="border-top: 5px solid #8b5cf6;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💰</div>
            <h2 style="color: #8b5cf6; margin: 0.5rem 0; font-size: 2.2rem; font-weight: 700;">${avg_revenue:.0f}</h2>
            <p style="color: #64748b; margin: 0; font-weight: 500;">Average Revenue Per User</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Business Intelligence Charts
    st.markdown('<div class="section-header"><h2>📈 Business Intelligence Analytics</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container-modern">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Customer Retention Overview")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = df['Churn'].value_counts()
        colors = ['#3b82f6', '#ef4444']
        
        wedges, texts, autotexts = ax.pie(
            churn_counts.values, 
            labels=['Retained', 'Churned'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 3}
        )
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('600')
            text.set_color('#1e293b')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax.set_title('Customer Distribution Analysis', fontsize=14, fontweight='700', color='#1e293b', pad=20)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container-modern">', unsafe_allow_html=True)
        st.markdown("#### 💵 Revenue Impact Analysis")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
        retained = df[df['Churn'] == 'No']['MonthlyCharges']
        
        bp = ax.boxplot([retained, churned], 
                       labels=['Retained', 'Churned'],
                       patch_artist=True,
                       boxprops={'alpha': 0.8, 'linewidth': 2},
                       medianprops={'color': 'white', 'linewidth': 3})
        
        bp['boxes'][0].set_facecolor('#3b82f6')
        bp['boxes'][1].set_facecolor('#ef4444')
        
        ax.set_title('Monthly Revenue Distribution', fontsize=14, fontweight='700', color='#1e293b', pad=20)
        ax.set_ylabel('Monthly Charges ($)', fontsize=12, color='#64748b')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('white')
        
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_interface(encoders, scaler, feature_names, models):
    """Modern prediction interface"""
    
    st.markdown('<div class="section-header"><h2>🔮 Customer Churn Risk Assessment</h2></div>', unsafe_allow_html=True)
    
    with st.form("customer_prediction"):
        st.markdown("### 📋 Customer Profile Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Account Information**")
            tenure = st.number_input("Account Tenure (months)", 0, 72, 12, help="Customer relationship duration")
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col2:
            st.markdown("**👤 Demographics & Services**")
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 
                                          value=monthly_charges * tenure,
                                          help="Cumulative charges to date")
        
        submitted = st.form_submit_button("🔍 Analyze Churn Risk", use_container_width=True)
        
        if submitted:
            risk_score = calculate_churn_risk(tenure, monthly_charges, contract, 
                                            payment_method, senior_citizen, internet_service)
            display_professional_results(risk_score, monthly_charges, tenure)

def calculate_churn_risk(tenure, monthly_charges, contract, payment_method, senior_citizen, internet_service):
    """Advanced risk calculation algorithm"""
    base_risk = 0.15
    
    contract_risk = {"Month-to-month": 0.35, "One year": 0.10, "Two year": 0.05}
    base_risk += contract_risk.get(contract, 0.10)
    
    if tenure <= 6:
        base_risk += 0.25
    elif tenure <= 12:
        base_risk += 0.15
    elif tenure > 36:
        base_risk -= 0.10
    
    if payment_method == "Electronic check":
        base_risk += 0.12
    elif payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        base_risk -= 0.05
    
    if monthly_charges > 80:
        base_risk += 0.08
    elif monthly_charges < 30:
        base_risk -= 0.03
    
    if senior_citizen == "Yes":
        base_risk += 0.04
    
    if internet_service == "Fiber optic":
        base_risk += 0.03
    
    return min(max(base_risk, 0.01), 0.95)

def display_professional_results(risk_score, monthly_charges, tenure):
    """Display modern prediction results"""
    
    if risk_score > 0.65:
        risk_level = "HIGH"
        css_class = "prediction-high-risk"
    elif risk_score > 0.35:
        risk_level = "MEDIUM"
        css_class = "prediction-medium-risk"
    else:
        risk_level = "LOW"
        css_class = "prediction-low-risk"
    
    st.markdown(f'''
    <div class="{css_class}">
        <h2>{risk_level} CHURN RISK</h2>
        <h1 style="font-size: 3.5rem; margin: 1rem 0;">{risk_score:.0%}</h1>
        <p style="font-size: 1.1rem;">Probability of customer churn</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        annual_value = monthly_charges * 12
        st.metric("Annual Customer Value", f"${annual_value:,.0f}")
    
    with col2:
        lifetime_value = monthly_charges * tenure
        st.metric("Lifetime Value to Date", f"${lifetime_value:,.0f}")
    
    with col3:
        retention_cost = annual_value * 0.15
        st.metric("Est. Retention Cost", f"${retention_cost:,.0f}")

def show_analytics_dashboard(df):
    """Modern analytics dashboard"""
    
    st.markdown('<div class="section-header"><h2>📊 Business Analytics Center</h2></div>', unsafe_allow_html=True)
    
    analysis_type = st.selectbox(
        "Select Analysis Focus:",
        ["📋 Contract Analysis", "💳 Payment Methods", "⏰ Customer Lifecycle"],
        index=0
    )
    
    st.markdown('<div class="chart-container-modern">', unsafe_allow_html=True)
    
    if analysis_type == "📋 Contract Analysis":
        st.markdown("### Contract Type Impact Analysis")
        
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        contract_churn.plot(kind='bar', ax=ax, color=['#3b82f6', '#ef4444'], alpha=0.8)
        ax.set_title('Churn Rate by Contract Type', fontsize=16, fontweight='700', pad=20)
        ax.set_xlabel('Contract Type', fontsize=12)
        ax.set_ylabel('Churn Rate', fontsize=12)
        ax.legend(['Retained', 'Churned'], fontsize=11)
        plt.xticks(rotation=0, fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8fafc')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown("#### 💡 Key Insights:")
        for contract in contract_churn.index:
            churn_rate = contract_churn.loc[contract, 'Yes']
            customer_count = len(df[df['Contract'] == contract])
            risk_indicator = "🔴" if churn_rate > 0.4 else "🟡" if churn_rate > 0.2 else "🟢"
            st.markdown(f"{risk_indicator} **{contract}**: {churn_rate:.1%} churn rate ({customer_count:,} customers)")
    
    elif analysis_type == "💳 Payment Methods":
        st.markdown("### Payment Method Risk Analysis")
        
        payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        payment_churn.plot(kind='bar', ax=ax, color=['#3b82f6', '#ef4444'], alpha=0.8)
        ax.set_title('Churn Rate by Payment Method', fontsize=16, fontweight='700', pad=20)
        ax.set_xlabel('Payment Method', fontsize=12)
        ax.set_ylabel('Churn Rate', fontsize=12)
        ax.legend(['Retained', 'Churned'], fontsize=11)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8fafc')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    elif analysis_type == "⏰ Customer Lifecycle":
        st.markdown("### Customer Lifecycle Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        churned_tenure = df[df['Churn'] == 'Yes']['tenure']
        retained_tenure = df[df['Churn'] == 'No']['tenure']
        
        ax.hist([retained_tenure, churned_tenure], bins=24, alpha=0.8, 
                color=['#3b82f6', '#ef4444'], label=['Retained', 'Churned'])
        ax.set_title('Customer Tenure Distribution', fontsize=16, fontweight='700', pad=20)
        ax.set_xlabel('Tenure (months)', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=12)
        ax.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8fafc')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_comparison(models):
    """Enhanced model comparison page"""
    
    st.markdown('<div class="section-header"><h2>🤖 Model Performance Center</h2></div>', unsafe_allow_html=True)
    
    try:
        test_data = pd.read_csv('data/processed/test_data.csv')
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        if models is None or len(models) == 0:
            st.warning("No trained models found. Please train models first.")
            return
        
        model_results = evaluate_all_models(models, X_test, y_test)
        
        if not model_results:
            st.error("Could not evaluate models. Please check model files.")
            return
        
        display_model_performance_table(model_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container-modern">', unsafe_allow_html=True)
            display_performance_chart(model_results)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container-modern">', unsafe_allow_html=True)
            display_roc_curves(model_results, y_test)
            st.markdown('</div>', unsafe_allow_html=True)
        
        show_best_model_summary(model_results)
        
    except FileNotFoundError:
        st.error("Test data not found. Please run the data processing pipeline first.")

def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models and return performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        except Exception as e:
            st.warning(f"Error evaluating {name}: {str(e)}")
            continue
    
    return results

def display_model_performance_table(model_results):
    """Display professional performance metrics table"""
    
    st.subheader("📊 Performance Metrics")
    
    comparison_data = []
    for name, metrics in model_results.items():
        comparison_data.append({
            'Model': name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1_score']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    best_model = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc'])
    best_auc = model_results[best_model]['roc_auc']
    
    st.success(f"🏆 Best Performing Model: **{best_model.replace('_', ' ').title()}** (ROC-AUC: {best_auc:.3f})")

def display_performance_chart(model_results):
    """Display enhanced performance comparison chart"""
    
    st.subheader("📈 Performance Comparison")
    
    models = [name.replace('_', ' ').title() for name in model_results.keys()]
    accuracies = [model_results[name]['accuracy'] for name in model_results.keys()]
    roc_aucs = [model_results[name]['roc_auc'] for name in model_results.keys()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='#3b82f6')
    bars2 = ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8, color='#ef4444')
    
    ax.set_xlabel('Models', fontweight='600', fontsize=12)
    ax.set_ylabel('Score', fontweight='600', fontsize=12)
    ax.set_title('Model Performance Comparison', fontweight='700', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='600')
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8fafc')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

def display_roc_curves(model_results, y_test):
    """Display enhanced ROC curves"""
    from sklearn.metrics import roc_curve
    
    st.subheader("📊 ROC Analysis")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
    
    for i, (name, metrics) in enumerate(model_results.items()):
        fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])
        auc_score = metrics['roc_auc']
        color = colors[i % len(colors)]
        
        ax.plot(fpr, tpr, color=color, linewidth=3, 
               label=f'{name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontweight='600', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='600', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontweight='700', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8fafc')
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

def show_best_model_summary(model_results):
    """Show enhanced best model summary"""
    
    st.markdown('<div class="section-header"><h3>🏆 Best Model Analysis</h3></div>', unsafe_allow_html=True)
    
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc'])
    best_metrics = model_results[best_model_name]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", best_model_name.replace('_', ' ').title())
        st.metric("ROC-AUC Score", f"{best_metrics['roc_auc']:.3f}")
    
    with col2:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{best_metrics['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{best_metrics['recall']:.3f}")
        st.metric("F1-Score", f"{best_metrics['f1_score']:.3f}")

def show_about_system():
    """Professional about page with proper Streamlit formatting"""
    
    st.markdown("# ℹ️ System Overview")
    
    st.markdown("""
    ## Customer Churn Prediction Analytics Platform
    
    This enterprise-grade analytics platform leverages advanced machine learning algorithms 
    to predict customer churn and provide actionable business intelligence.
    """)
    
    # Core Capabilities Section
    st.markdown("### 🎯 Core Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🧠 Predictive Intelligence**  
        Real-time churn probability assessment with advanced machine learning algorithms
        
        **📊 Executive Dashboards**  
        C-suite appropriate analytics and strategic business intelligence
        
        **🎯 Risk Segmentation**  
        Automated customer risk categorization and targeted intervention strategies
        """)
    
    with col2:
        st.markdown("""
        **💰 Financial Impact**  
        Revenue-at-risk calculations and comprehensive ROI modeling
        
        **🎯 Strategic Recommendations**  
        Actionable retention protocols and data-driven decision support
        """)
    
    # Technical Architecture Section
    st.markdown("### 🏗️ Technical Architecture")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("""
        **🤖 Machine Learning**
        - Ensemble Models
        - XGBoost
        - Scikit-learn
        """)
    
    with col2:
        st.success("""
        **🔧 Data Engineering**
        - Automated ETL
        - Feature Engineering
        - Pandas
        """)
    
    with col3:
        st.error("""
        **🎨 User Experience**
        - Streamlit
        - Responsive UI
        - Modern CSS
        """)
    
    with col4:
        st.warning("""
        **📊 Visualization**
        - Matplotlib
        - Interactive Charts
        - Business Intelligence
        """)
    
    # Business Value Section
    st.markdown("### 💼 Business Value")
    
    st.info("""
    **Built for Enterprise Customer Success Teams**
    
    Empowering data-driven retention strategies through advanced predictive analytics 
    and actionable business intelligence.
    """)
    
    # Technical Details
    st.markdown("### 🛠️ Technical Stack")
    
    tech_stack = {
        "Language": "Python 3.7+",
        "ML Framework": "Scikit-learn, XGBoost",
        "Web Framework": "Streamlit",
        "Data Processing": "Pandas, NumPy",
        "Visualization": "Matplotlib, Seaborn",
        "Model Persistence": "Joblib"
    }
    
    for tech, desc in tech_stack.items():
        st.markdown(f"**{tech}**: {desc}")
    
    # Developer Information
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        **Ankur Yadav**
        
        Machine Learning Engineer & Data Scientist
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/ankur-yadav-0403bb2a9)")
        with col_b:
            st.markdown("[🐙 GitHub](https://github.com/incendio221)")
    
      


if __name__ == "__main__":
    main()
