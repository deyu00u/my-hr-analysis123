import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="HR Attrition Project", layout="wide")

def clean_data(df):
    df.columns = df.columns.astype(str).str.strip().str.replace('ï»¿', '')
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).str.replace('ï»¿', '')
    return df

st.title("HR Attrition: Data Science Lifecycle")
st.markdown("---")

file_name = "HR-Attrition.csv"

if os.path.exists(file_name):
    try:
        df_raw = pd.read_csv(file_name, sep=',', engine='python', encoding='latin1', on_bad_lines='skip')
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
            st.write("1. Encoding: Standardized character formatting to ensure data integrity.")
            st.write("2. Feature Filtering: Removed constant or non-predictive variables like EmployeeNumber and StandardHours.")
            st.write("3. Categorical Conversion: Used Label Encoding to prepare text data for the Random Forest algorithm.")
            
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.markdown("---")
            st.write("Dataset Preview:")
            st.dataframe(df_clean.head(10))
            st.info(f"Total Rows Processed: {len(df_clean)}")
        
        with tab3:
            st.header("Phase 4: Exploratory Data Analysis")
            st.write("Analysis of the relationship between numerical attributes and Attrition status.")
            
            df_corr = df_clean.copy()
            if 'Attrition' in df_corr.columns:
                df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
                
                numeric_df = df_corr.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    corr_series = numeric_df.corr()['Attrition'].sort_values()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr_series.plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_title("Feature Correlation with Attrition")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.error("Attrition column not found in data.")

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
                
                st.metric("Model Prediction Accuracy", f"{accuracy*100:.2f}%")
                
                st.subheader("Key Predictive Factors")
                st.write("The model identified the following features as most impactful:")
                importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                importances.plot(kind='barh', color='seagreen', ax=ax_imp)
                plt.tight_layout()
                st.pyplot(fig_imp)
            except Exception as e:
                st.error(f"Modeling error: {e}")

        with tab5:
            st.header("Phase 6: Prediction Tool")
            st.write("Input employee details to evaluate potential attrition risk.")
            
            col1, col2 = st.columns(2)
            with col1:
                input_age = st.slider("Age", 18, 60, 30)
                input_overtime = st.selectbox("Overtime Status", ["Yes", "No"])
            with col2:
                input_income = st.number_input("Monthly Income", value=5000)
                input_sat = st.slider("Job Satisfaction Rating (1-4)", 1, 4, 3)
            
            risk_score = 0
            if input_overtime == "Yes": risk_score += 40
            if input_income < 3500: risk_score += 30
            if input_age < 25: risk_score += 15
            if input_sat < 2: risk_score += 15
            
            st.markdown("---")
            if risk_score >= 50:
                st.error(f"Assessment: High Attrition Risk ({risk_score}%)")
            else:
                st.success(f"Assessment: Low Attrition Risk ({risk_score}%)")

    except Exception as e:
        st.error(f"Error reading dataset: {e}")
else:
    st.warning("Data file not found. Please verify that 'HR-Attrition.csv' is present in the repository.")
