import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# Set page to wide mode for better visibility
st.set_page_config(page_title="HR Attrition Analysis Pro", layout="wide")

# Helper function to clean text strings from any hidden artifacts
def clean_text(text):
    if isinstance(text, str):
        # Removes non-ASCII characters like the ï»¿ artifact
        return re.sub(r'[^\x00-\x7F]+', '', text).strip()
    return text

st.title("📊 HR Attrition: Data Science Lifecycle")
st.markdown("---")

# --- PHASE 2: DATA ACQUISITION ---
st.sidebar.header("Phase 2: Data Acquisition")
uploaded_file = st.sidebar.file_uploader("Upload HR-Attrition.csv", type="csv")

if uploaded_file is not None:
    # 'utf-8-sig' specifically fixes the ï»¿ gibberish issue
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    
    # Apply cleaning to headers and object columns just in case
    df.columns = [clean_text(col) for col in df.columns]
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(clean_text)

    # Create Tabs for the 6 Phases (Combining 2 into the sidebar/main)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Phase 1: Business", 
        "Phase 3: Preparation", 
        "Phase 4: Exploration", 
        "Phase 5: Modeling", 
        "Phase 6: Deployment"
    ])

    with tab1:
        st.header("Phase 1: Business Understanding")
        st.subheader("Project Objectives")
        st.write("""
        Employee attrition represents a significant cost to organizations in terms of recruitment, 
        training, and lost productivity. The goal of this analysis is to:
        """)
        st.info("""
        1. **Identify Key Drivers:** Determine which factors (e.g., Pay, Overtime, Age) contribute most to turnover.
        2. **Risk Assessment:** Build a predictive model to identify employees at high risk of leaving.
        3. **Strategic Retention:** Provide data-driven insights to HR to improve employee satisfaction and ROI.
        """)

    with tab2:
        st.header("Phase 3: Data Preparation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Cleaning:** Dropping non-predictive variables (Employee ID, etc.)")
            # Dropping irrelevant columns
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.success("Constant and Redundant features removed.")
        
        with col2:
            st.write("**Dataset Preview:**")
            st.dataframe(df_clean.head(10))

    with tab3:
        st.header("Phase 4: Exploratory Data Analysis")
        st.write("Correlation analysis showing how different features relate to Attrition.")
        
        df_corr = df_clean.copy()
        # Convert Target and Overtime to numeric for correlation
        df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
        # Select numeric columns for correlation
        numeric_cols = df_corr.select_dtypes(include=['number'])
        corr = numeric_cols.corr()['Attrition'].sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if x > 0 else 'steelblue' for x in corr]
        corr.drop('Attrition').plot(kind='barh', ax=ax, color=colors)
        ax.set_title("Feature Correlation with Employee Attrition")
        st.pyplot(fig)

    with tab4:
        st.header("Phase 5: Modeling & Evaluation")
        
        # Prepare Data for Machine Learning
        le = LabelEncoder()
        df_ml = df_clean.copy()
        for col in df_ml.select_dtypes(include=['object']).columns:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        
        X = df_ml.drop('Attrition', axis=1)
        y = df_ml['Attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        accuracy = rf.score(X_test, y_test)
        
        st.metric("Model Prediction Accuracy", f"{accuracy*100:.2f}%")
        
        # Feature Importance
        st.subheader("Key Drivers Identified by AI")
        importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10).sort_values()
        fig_imp, ax_imp = plt.subplots()
        importances.plot(kind='barh', color='seagreen', ax=ax_imp)
        st.pyplot(fig_imp)

    with tab5:
        st.header("Phase 6: Deployment (Prediction Tool)")
        st.write("Simulate an employee profile to predict attrition risk.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 30)
            monthly_income = st.number_input("Monthly Income ($)", value=5000)
        with c2:
            overtime = st.selectbox("Working Overtime?", ["Yes", "No"])
            job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        with c3:
            distance = st.slider("Distance from Home (km)", 1, 30, 10)
            stock = st.slider("Stock Option Level", 0, 3, 1)

        # Basic logic for the "Tool" simulation
        st.markdown("---")
        risk_score = 0
        if overtime == "Yes": risk_score += 40
        if monthly_income < 3000: risk_score += 30
        if job_sat < 2: risk_score += 20
        if distance > 20: risk_score += 10
        
        if risk_score >= 50:
            st.error(f"Prediction: High Risk of Attrition ({risk_score}%)")
        else:
            st.success(f"Prediction: Low Risk of Attrition ({risk_score}%)")

else:
    # Professional Landing Page
    st.info("👋 Welcome! Please upload the **HR-Attrition.csv** file in the sidebar to begin the analysis.")
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&q=80&w=2070", use_column_width=True)
