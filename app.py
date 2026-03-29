import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import re

st.set_page_config(page_title="HR Attrition Analysis", layout="wide")

def clean_data(df):
    df.columns = [re.sub(r'[^\x00-\x7F]+', '', str(col)).strip() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('ï»¿', '', regex=False).str.strip()
            df[col] = df[col].apply(lambda x: re.sub(r'[^\x20-\x7E]+', '', x))
    return df

st.title("HR Attrition Analysis: Data Science Lifecycle")
st.markdown("---")

file_name = "HR-Attrition.csv"

if os.path.exists(file_name):
    try:
        df_raw = pd.read_csv(file_name, engine='python', encoding='latin1', on_bad_lines='skip')
        df = clean_data(df_raw)
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Phase 1: Business", 
            "Phase 2: Acquisition",
            "Phase 3: Preparation", 
            "Phase 4: Exploration", 
            "Phase 5: Modeling", 
            "Phase 6: Prediction"
        ])

        with tab1:
            st.header("Phase 1: Business Understanding")
            st.write("The objective is to identify key drivers of employee turnover. By understanding why staff leave, the organization can reduce replacement costs and improve long-term retention.")
        
        with tab2:
            st.header("Phase 2: Data Acquisition")
            st.write("The dataset used for this analysis is the HR-Attrition source file, containing 1,470 employee records and 35 initial features including demographics, job roles, and satisfaction levels.")
            st.info(f"Source file '{file_name}' loaded successfully.")

        with tab3:
            st.header("Phase 3: Data Preparation")
            st.subheader("Data Cleaning and Feature Selection")
            st.write("1. Data Cleaning: Handled character encoding and stripped non-printable artifacts from all text fields.")
            st.write("2. Redundancy Filter: Removed features with zero variance or no predictive value (EmployeeCount, StandardHours, Over18, EmployeeNumber).")
            st.write("3. Transformation: Formatted categorical strings for downstream machine learning processing.")
            
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.markdown("---")
            st.write("Cleaned Dataset Sample (Pre-Encoding):")
            st.dataframe(df_clean.head(10))

        with tab4:
            st.header("Phase 4: Exploratory Data Analysis")
            st.write("Statistical correlation of all features (including categorical) against the Attrition target.")
            
            df_eda = df_clean.copy()
            le_eda = LabelEncoder()
            for col in df_eda.select_dtypes(include=['object']).columns:
                df_eda[col] = le_eda.fit_transform(df_eda[col].astype(str))
            
            if 'Attrition' in df_eda.columns:
                corr_series = df_eda.corr()['Attrition'].sort_values()
                corr_series = corr_series.drop('Attrition', errors='ignore')
                
                fig, ax = plt.subplots(figsize=(10, 10))
                corr_series.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title("Feature Correlation with Attrition")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("Attrition column missing.")

        with tab5:
            st.header("Phase 5: Modeling and Evaluation")
            le = LabelEncoder()
            df_ml = df_clean.copy()
            
            # Map for prediction logic later
            label_mappings = {}
            for col in df_ml.select_dtypes(include=['object']).columns:
                label_mappings[col] = list(le.fit(df_ml[col].astype(str)).classes_)
                df_ml[col] = le.transform(df_ml[col].astype(str))
            
            X = df_ml.drop('Attrition', axis=1)
            y = df_ml['Attrition']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            
            st.metric("Model Prediction Accuracy", f"{accuracy*100:.2f}%")
            
            st.subheader("Top Predictive Features")
            importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            importances.plot(kind='barh', color='seagreen', ax=ax_imp)
            plt.tight_layout()
            st.pyplot(fig_imp)

        with tab6:
            st.header("Phase 6: Prediction Tool")
            st.write("Use the trained Random Forest model to predict risk based on a specific profile.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                age_in = st.slider("Age", 18, 60, 30)
                income_in = st.number_input("Monthly Income", value=5000)
            with col2:
                ot_in = st.selectbox("Overtime", ["Yes", "No"])
                dist_in = st.slider("Distance From Home", 1, 30, 10)
            with col3:
                sat_in = st.slider("Job Satisfaction", 1, 4, 3)
                env_in = st.slider("Environment Satisfaction", 1, 4, 3)

            # Create a sample input matching the training features
            sample_data = X_train.iloc[0:1].copy()
            for col in sample_data.columns:
                if col == 'Age': sample_data[col] = age_in
                elif col == 'MonthlyIncome': sample_data[col] = income_in
                elif col == 'OverTime': sample_data[col] = 1 if ot_in == "Yes" else 0
                elif col == 'DistanceFromHome': sample_data[col] = dist_in
                elif col == 'JobSatisfaction': sample_data[col] = sat_in
                elif col == 'EnvironmentSatisfaction': sample_data[col] = env_in

            prediction_prob = model.predict_proba(sample_data)[0][1]
            st.markdown("---")
            if prediction_prob > 0.5:
                st.error(f"Prediction: High Risk ({prediction_prob*100:.1f}%)")
            else:
                st.success(f"Prediction: Low Risk ({prediction_prob*100:.1f}%)")

    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.warning("File 'HR-Attrition.csv' not found.")
    
