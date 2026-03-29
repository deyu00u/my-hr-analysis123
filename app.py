import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import re

st.set_page_config(page_title="HR Attrition Project", layout="wide")

def clean_data(df):
    # Fix headers: Remove non-ASCII characters and hidden symbols
    df.columns = [re.sub(r'[^\x00-\x7F]+', '', str(col)).strip() for col in df.columns]
    
    # Fix cell values: Specifically target the 'ï»¿' and other artifacts in text
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('ï»¿', '', regex=False).str.strip()
            # Remove any remaining non-printable characters
            df[col] = df[col].apply(lambda x: re.sub(r'[^\x20-\x7E]+', '', x))
            
    return df

st.title("HR Attrition Analysis: Data Science Lifecycle")
st.markdown("---")

file_name = "HR-Attrition.csv"

if os.path.exists(file_name):
    try:
        # Using python engine to prevent buffer errors
        df_raw = pd.read_csv(file_name, engine='python', encoding='latin1', on_bad_lines='skip')
        df = clean_data(df_raw)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Business Understanding", 
            "Data Preparation", 
            "Exploratory Analysis", 
            "Modeling", 
            "Prediction Tool"
        ])

        with tab1:
            st.header("Phase 1: Business Understanding")
            st.write("The objective of this project is to analyze the primary factors leading to employee attrition. By identifying these trends, the company can implement targeted retention programs to reduce turnover costs.")
        
        with tab2:
            st.header("Phase 3: Data Preparation")
            st.subheader("Processing Steps")
            st.write("1. Data Cleaning: Handled character encoding and standardized all column headers for consistency.")
            st.write("2. Feature Engineering: Removed constant variables that do not contribute to model variance.")
            st.write("3. Categorical Encoding: Converted textual data into numerical format for algorithmic processing.")
            
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.markdown("---")
            st.write("Cleaned Dataset Sample:")
            st.dataframe(df_clean.head(10))
            st.info(f"Total Observations: {len(df_clean)}")
        
        with tab3:
            st.header("Phase 4: Exploratory Data Analysis")
            st.write("Correlation analysis showing the relationship between employee features and Attrition.")
            
            df_corr = df_clean.copy()
            if 'Attrition' in df_corr.columns:
                df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
                
                numeric_df = df_corr.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    corr_series = numeric_df.corr()['Attrition'].sort_values()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr_series.plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_title("Feature Correlation with Attrition Status")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.error("Target variable 'Attrition' not found.")

        with tab4:
            st.header("Phase 5: Modeling and Evaluation")
            try:
                le = LabelEncoder()
                df_ml = df_clean.copy()
                df_ml['Attrition'] = df_ml['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

                for col in df_ml.select_dtypes(include=['object']).columns:
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                
                X = df_ml.drop('Attrition', axis=1)
                y = df_ml['Attrition']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                
                st.metric("Model Performance (Accuracy)", f"{accuracy*100:.2f}%")
                
                st.subheader("Predictive Feature Importance")
                importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                importances.plot(kind='barh', color='seagreen', ax=ax_imp)
                plt.tight_layout()
                st.pyplot(fig_imp)
            except Exception as e:
                st.error(f"Error during model execution: {e}")

        with tab5:
            st.header("Phase 6: Prediction Tool")
            st.write("Simulate employee profiles to evaluate the calculated attrition risk.")
            
            col1, col2 = st.columns(2)
            with col1:
                input_age = st.slider("Age", 18, 60, 30)
                input_overtime = st.selectbox("Overtime Status", ["Yes", "No"])
            with col2:
                input_income = st.number_input("Monthly Income", value=5000)
                input_sat = st.slider("Job Satisfaction Rating (1-4)", 1, 4, 3)
            
            score = 0
            if input_overtime == "Yes": score += 40
            if input_income < 3500: score += 30
            if input_age < 25: score += 15
            if input_sat < 2: score += 15
            
            st.markdown("---")
            if score >= 50:
                st.error(f"Calculated Risk: High Attrition Risk ({score}%)")
            else:
                st.success(f"Calculated Risk: Low Attrition Risk ({score}%)")

    except Exception as e:
        st.error(f"Dataset access error: {e}")
else:
    st.warning("The source file 'HR-Attrition.csv' was not found in the directory.")
