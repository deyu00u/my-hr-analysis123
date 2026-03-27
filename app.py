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
uploaded_file = st.sidebar.file_uploader("Upload HR Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # --- SMART FILE READING ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Check if Attrition exists, if not, try to find a similar column
        if 'Attrition' not in df.columns:
            st.error("Error: Could not find a column named 'Attrition'. Please check your data headers.")
            st.stop()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Business Understanding", "3. Data Preparation", "4. Exploratory Analysis", "5. Modeling", "6. Prediction Tool"])

        with tab1:
            st.header("Phase 1: Business Understanding")
            st.info("Objective: Identify predictors that correlate with employee attrition to improve retention.")
        
        with tab2:
            st.header("Phase 3: Data Preparation")
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.write("First 5 rows of your cleaned data:")
            st.dataframe(df_clean.head())
            st.success(f"Successfully Loaded: {len(df_clean)} rows.")
        
        with tab3:
            st.header("Phase 4: Exploratory Data Analysis")
            df_corr = df_clean.copy()
            # Convert Attrition to 1/0 for math
            df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
            numeric_df = df_corr.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.subheader("Correlation with Attrition")
                corr = numeric_df.corr()['Attrition'].sort_values()
                fig, ax = plt.subplots()
                corr.plot(kind='barh', ax=ax, color='skyblue')
                st.pyplot(fig)
            else:
                st.warning("No numeric data found for correlation analysis.")

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
                st.metric("Model Prediction Accuracy", f"{acc*100:.2f}%")
                
                st.subheader("Top Factors Influencing Attrition")
                feat_importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
                fig_feat, ax_feat = plt.subplots()
                feat_importances.sort_values().plot(kind='barh', color='lightgreen', ax=ax_feat)
                st.pyplot(fig_feat)
            except Exception as e:
                st.error(f"Modeling Error: {e}")

        with tab5:
            st.header("Phase 6: Deployment (Interactive Tool)")
            st.write("Predict the risk level for a specific employee:")
            col1, col2 = st.columns(2)
            with col1:
                age_val = st.slider("Employee Age", 18, 65, 30)
                ot_val = st.selectbox("Works Overtime?", ["Yes", "No"])
            with col2:
                inc_val = st.number_input("Monthly Income ($)", value=5000)
                sat_val = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                
            # Demo prediction logic
            score = 0
            if ot_val == "Yes": score += 45
            if inc_val < 4000: score += 30
            if age_val < 26: score += 15
            if sat_val < 2: score += 10
            
            if score >= 50:
                st.error(f"Prediction: High Risk ({score}%) - This employee is likely to leave.")
            else:
                st.success(f"Prediction: Low Risk ({score}%) - This employee is likely to stay.")

    except Exception as e:
        st.error(f"Critical File Error: {e}")
        st.write("Please ensure you are uploading a valid CSV or Excel file.")

else:
    st.info("Waiting for data... Please upload your file using the sidebar.")
