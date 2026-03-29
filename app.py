import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import re

# Page Config
st.set_page_config(page_title="HR Attrition Analysis", layout="wide")

# Aggressive Cleaning Function to remove 'gibberish' (BOM/Non-ASCII)
def fix_encoding_issues(df):
    # Clean Column Headers
    df.columns = [re.sub(r'[^\x00-\x7F]+', '', str(col)).strip() for col in df.columns]
    # Clean Cell Values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('ï»¿', '', regex=False).strip()
            df[col] = df[col].apply(lambda x: re.sub(r'[^\x20-\x7E]+', '', x))
    return df

st.title("HR Attrition Analysis: Data Science Lifecycle")
st.markdown("---")

file_name = "HR-Attrition.csv"

if os.path.exists(file_name):
    try:
        # engine='python' prevents the 'Buffer Overflow' error
        # encoding='latin1' handles the special characters
        df_raw = pd.read_csv(file_name, engine='python', encoding='latin1', on_bad_lines='skip')
        df = fix_encoding_issues(df_raw)
        
        # UI: 6 Phases of Data Science Lifecycle
        tabs = st.tabs([
            "Phase 1: Business", 
            "Phase 2: Acquisition",
            "Phase 3: Preparation", 
            "Phase 4: Exploration", 
            "Phase 5: Modeling", 
            "Phase 6: Prediction"
        ])

        with tabs[0]:
            st.header("Phase 1: Business Understanding")
            st.write("Goal: Identify the key drivers behind employee turnover to help HR implement better retention strategies.")
        
        with tabs[1]:
            st.header("Phase 2: Data Acquisition")
            st.write(f"Dataset '{file_name}' loaded successfully.")
            st.write(f"Total Records: {df.shape[0]} | Total Features: {df.shape[1]}")

        with tabs[2]:
            st.header("Phase 3: Data Preparation")
            st.write("Dropped non-predictive columns: EmployeeCount, StandardHours, Over18, EmployeeNumber.")
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.dataframe(df_clean.head(10))

        with tabs[3]:
            st.header("Phase 4: Exploratory Data Analysis")
            # Encode categories so they show up in the correlation chart
            df_eda = df_clean.copy()
            le_eda = LabelEncoder()
            for col in df_eda.select_dtypes(include=['object']).columns:
                df_eda[col] = le_eda.fit_transform(df_eda[col].astype(str))
            
            if 'Attrition' in df_eda.columns:
                corr = df_eda.corr()['Attrition'].sort_values().drop('Attrition', errors='ignore')
                fig, ax = plt.subplots(figsize=(10, 8))
                corr.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title("Correlation of Features with Attrition")
                plt.tight_layout()
                st.pyplot(fig)

        with tabs[4]:
            st.header("Phase 5: Modeling")
            # Prepare ML Model
            df_ml = df_clean.copy()
            le = LabelEncoder()
            for col in df_ml.select_dtypes(include=['object']).columns:
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            
            X = df_ml.drop('Attrition', axis=1)
            y = df_ml['Attrition']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            st.metric("Model Accuracy", f"{model.score(X_test, y_test)*100:.2f}%")
            
            # Feature Importance
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
            fig_imp, ax_imp = plt.subplots()
            feat_imp.plot(kind='barh', color='seagreen', ax=ax_imp)
            ax_imp.set_title("Top 10 Drivers of Attrition")
            st.pyplot(fig_imp)

        with tabs[5]:
            st.header("Phase 6: Prediction Tool")
            st.write("Adjust values to see the probability of an employee leaving.")
            
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 60, 30)
                income = st.number_input("Monthly Income", value=5000)
                overtime = st.selectbox("Overtime", ["Yes", "No"])
            with c2:
                dist = st.slider("Distance From Home", 1, 30, 10)
                job_sat = st.slider("Job Satisfaction", 1, 4, 3)
                env_sat = st.slider("Environment Satisfaction", 1, 4, 3)

            # Map inputs to model format
            input_row = X_train.iloc[0:1].copy()
            input_row.iloc[0] = X_train.median() # Fill others with median
            input_row['Age'] = age
            input_row['MonthlyIncome'] = income
            input_row['OverTime'] = 1 if overtime == "Yes" else 0
            input_row['DistanceFromHome'] = dist
            input_row['JobSatisfaction'] = job_sat
            input_row['EnvironmentSatisfaction'] = env_sat

            risk = model.predict_proba(input_row)[0][1]
            st.markdown("---")
            if risk > 0.5:
                st.error(f"High Risk of Attrition: {risk*100:.1f}%")
            else:
                st.success(f"Low Risk of Attrition: {risk*100:.1f}%")

    except Exception as e:
        st.error(f"Critical Error: {e}")
else:
    st.warning("Please ensure 'HR-Attrition.csv' is in your GitHub folder.")
