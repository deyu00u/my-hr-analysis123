import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re


st.set_page_config(page_title="HR Attrition Analysis Pro", layout="wide")


def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^\x00-\x7F]+', '', text).strip()
    return text

st.title("HR Attrition: Data Science Lifecycle")
st.markdown("---")


st.sidebar.header("Phase 2: Data Acquisition")
uploaded_file = st.sidebar.file_uploader("Upload HR-Attrition.csv", type="csv")

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    
    
    df.columns = [clean_text(col) for col in df.columns]
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(clean_text)


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
        st.write("Employee attrition causes significant operational costs. This project aims to:")
        st.write("1. **Identify Drivers:** Isolate variables like Overtime or Pay that lead to turnover.")
        st.write("2. **Risk Prediction:** Develop a model to flag high-risk employee profiles.")
        st.write("3. **Retention Strategy:** Provide data-driven insights for management intervention.")

    with tab2:
        st.header("Phase 3: Data Preparation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Cleaning:** Dropping non-predictive variables.")
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            
        with col2:
            st.write("**Dataset Preview:**")
            st.dataframe(df_clean.head(10))

    with tab3:
        st.header("Phase 4: Exploratory Data Analysis")
        st.write("Correlation analysis of all numeric features against Attrition.")
        
        df_corr = df_clean.copy()
        if 'Attrition' in df_corr.columns:
            df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
        numeric_cols = df_corr.select_dtypes(include=['number'])
        if not numeric_cols.empty and 'Attrition' in numeric_cols.columns:
            corr = numeric_cols.corr()['Attrition'].sort_values()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            corr.drop('Attrition', errors='ignore').plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title("Feature Correlation with Attrition")
            st.pyplot(fig)

    with tab4:
        st.header("Phase 5: Modeling & Evaluation")
        
        le = LabelEncoder()
        df_ml = df_clean.copy()
        for col in df_ml.select_dtypes(include=['object']).columns:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        
        if 'Attrition' in df_ml.columns:
            X = df_ml.drop('Attrition', axis=1)
            y = df_ml['Attrition']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            accuracy = rf.score(X_test, y_test)
            st.metric("Model Prediction Accuracy", f"{accuracy*100:.2f}%")
            
            st.subheader("Key Drivers Identified by Model")
            importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10).sort_values()
            fig_imp, ax_imp = plt.subplots()
            importances.plot(kind='barh', color='seagreen', ax=ax_imp)
            st.pyplot(fig_imp)

    with tab5:
        st.header("Phase 6: Deployment (Prediction Tool)")
        st.write("Simulate employee profiles to evaluate attrition risk.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 30)
            income = st.number_input("Monthly Income ($)", value=5000)
        with c2:
            overtime = st.selectbox("Working Overtime?", ["Yes", "No"])
            job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        with c3:
            distance = st.slider("Distance from Home (km)", 1, 30, 10)
            stock = st.slider("Stock Option Level", 0, 3, 1)

        st.markdown("---")
        risk_score = 0
        if overtime == "Yes": risk_score += 40
        if income < 3000: risk_score += 30
        if job_sat < 2: risk_score += 20
        if distance > 20: risk_score += 10
        
        st.subheader("Result:")
        if risk_score >= 50:
            st.write(f"**High Risk of Attrition ({risk_score}%)**")
        else:
            st.write(f"**Low Risk of Attrition ({risk_score}%)**")

else:
    st.write("Please upload the HR-Attrition.csv file in the sidebar to begin the analysis.")
