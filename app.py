import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="HR Attrition Analysis", layout="wide")

st.title("HR Attrition: Data Science Lifecycle")
st.markdown("---")

st.sidebar.header("Phase 2: Data Acquisition")
uploaded_file = st.sidebar.file_uploader("Upload HR-Attrition.csv", type="csv")

if uploaded_file is not None:
    # THE FIX: Added engine, sep=None, and encoding to handle all CSV types
    df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Business Understanding", "3. Data Preparation", "4. Exploratory Analysis", "5. Modeling", "6. Prediction Tool"])

    with tab1:
        st.header("Phase 1: Business Understanding")
        st.info("Objective: Identify predictors that correlate with employee attrition to improve retention.")
    
    with tab2:
        st.header("Phase 3: Data Preparation")
        # Remove unneeded columns if they exist
        df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
        st.dataframe(df_clean.head())
        st.success(f"Data Loaded: {len(df_clean)} rows.")
    
    with tab3:
        st.header("Phase 4: Exploratory Data Analysis")
        if 'Attrition' in df_clean.columns:
            df_corr = df_clean.copy()
            df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
            # Simple Correlation Plot
            numeric_df = df_corr.select_dtypes(include=['number'])
            if not numeric_df.empty:
                corr = numeric_df.corr()['Attrition'].sort_values()
                fig, ax = plt.subplots()
                corr.plot(kind='barh', ax=ax, color='skyblue')
                st.pyplot(fig)
        else:
            st.error("Column 'Attrition' not found in data.")

    with tab4:
        st.header("Phase 5: Modeling")
        try:
            le = LabelEncoder()
            df_ml = df_clean.copy()
            for col in df_ml.select_dtypes(include=['object']).columns:
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            
            X = df_ml.drop('Attrition', axis=1)
            y = df_ml['Attrition']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            acc = rf.score(X_test, y_test)
            st.metric("Model Accuracy", f"{acc*100:.2f}%")
            
            st.subheader("Top Predictors")
            feat_importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
            fig_feat, ax_feat = plt.subplots()
            feat_importances.sort_values().plot(kind='barh', color='lightgreen', ax=ax_feat)
            st.pyplot(fig_feat)
        except Exception as e:
            st.error(f"Error in Modeling: {e}")

    with tab5:
        st.header("Phase 6: Deployment")
        st.write("Test employee risk based on key factors:")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 60, 30)
            overtime = st.selectbox("Overtime", ["Yes", "No"])
        with c2:
            income = st.number_input("Monthly Income", value=5000)
            satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            
        # Basic Risk Logic for Demo
        risk = 0
        if overtime == "Yes": risk += 40
        if income < 3500: risk += 30
        if satisfaction < 2: risk += 20
        
        if risk >= 50:
            st.error(f"Risk Score: {risk}% - High Attrition Risk")
        else:
            st.success(f"Risk Score: {risk}% - Low Attrition Risk")

else:
    st.warning("Please upload the HR-Attrition.csv file in the sidebar to begin.")
