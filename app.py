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
....df = pd.read_csv(uploaded_file, encoding='latin1')
....tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Business Understanding", "3. Data Preparation", "4. Exploratory Analysis", "5. Modeling", "6. Prediction Tool"])
    with tab1:
        st.header("Phase 1: Business Understanding")
        st.info("Objective: Identify predictors that correlate with employee attrition to improve retention.")
    
    with tab2:
        st.header("Phase 3: Data Preparation")
        df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
        st.dataframe(df_clean.head())
    
    with tab3:
        st.header("Phase 4: Exploratory Data Analysis")
        df_corr = df_clean.copy()
        df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
        df_corr['OverTime'] = df_corr['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
        corr = df_corr.select_dtypes(include=['number']).corr()['Attrition'].sort_values()
        fig, ax = plt.subplots()
        corr.plot(kind='barh', ax=ax)
        st.pyplot(fig)

    with tab4:
        st.header("Phase 5: Modeling")
        le = LabelEncoder()
        df_ml = df_clean.copy()
        for col in df_ml.select_dtypes(include=['object']).columns:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        X = df_ml.drop('Attrition', axis=1)
        y = df_ml['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        st.metric("Model Accuracy", f"{rf.score(X_test, y_test)*100:.2f}%")

    with tab5:
        st.header("Phase 6: Deployment")
        st.write("Use the sidebar to upload data and test the model.")
else:
    st.warning("Please upload the HR-Attrition.csv file to begin.")
