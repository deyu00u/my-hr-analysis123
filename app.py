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
    df_raw = pd.read_csv(file_name, sep=None, engine='python', encoding='latin1')
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
        st.write("The objective of this study is to analyze employee turnover. By identifying key drivers of attrition, the organization can implement strategic changes to improve employee retention and reduce hiring costs.")
    
    with tab2:
        st.header("Phase 3: Data Preparation")
        st.subheader("Data Cleaning Steps")
        st.write("1. Standardized character encoding for all column headers.")
        st.write("2. Removed non-predictive features: EmployeeCount, StandardHours, Over18, and EmployeeNumber.")
        st.write("3. Handled categorical variables using label encoding for machine learning compatibility.")
        
      
        df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
        st.markdown("---")
        st.write("Dataset Preview:")
        st.dataframe(df_clean.head(10))
        st.info(f"Total Records Processed: {len(df_clean)}")
    
    with tab3:
        st.header("Phase 4: Exploratory Data Analysis")
        st.write("Correlation analysis showing the relationship between numeric features and Attrition.")
        
        df_corr = df_clean.copy()
        if 'Attrition' in df_corr.columns:
           
            df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
            numeric_df = df_corr.select_dtypes(include=['number'])
            if not numeric_df.empty:
                
                corr_series = numeric_df.corr()['Attrition'].sort_values()
                
                
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_series.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title("Correlation of Features with Attrition")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.error("Target column 'Attrition' not detected.")

    with tab4:
        st.header("Phase 5: Modeling and Evaluation")
        try:
            # Prepare data for Random Forest
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
            
            st.metric("Model Accuracy Score", f"{accuracy*100:.2f}%")
            
            st.subheader("Feature Importance")
            st.write("The following features were identified as the strongest predictors of attrition:")
            importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(10).sort_values()
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            importances.plot(kind='barh', color='seagreen', ax=ax_imp)
            plt.tight_layout()
            st.pyplot(fig_imp)
        except Exception as e:
            st.error(f"Error during modeling: {e}")

    with tab5:
        st.header("Phase 6: Prediction Tool")
        st.write("Interactive simulation to predict attrition risk for a single employee profile.")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Employee Age", 18, 60, 30)
            overtime = st.selectbox("Working Overtime", ["Yes", "No"])
        with col2:
            income = st.number_input("Monthly Income (USD)", value=5000)
            satisfaction = st.slider("Job Satisfaction Level (1-4)", 1, 4, 3)
        
       
        score = 0
        if overtime == "Yes": score += 40
        if income < 3500: score += 30
        if age < 25: score += 15
        if satisfaction < 2: score += 15
        
        st.markdown("---")
        if score >= 50:
            st.error(f"Result: High Risk of Attrition ({score}%)")
        else:
            st.success(f"Result: Low Risk of Attrition ({score}%)")

else:
    st.warning("Dataset not found. Please ensure 'HR-Attrition.csv' is uploaded to the repository.")
