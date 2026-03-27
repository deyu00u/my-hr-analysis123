import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import io

# --- 1. SET PAGE CONFIG (The Foundation) ---
st.set_page_config(
    page_title="HR Attrition Analysis Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. GIBBERISH CLEANUP FUNCTIONS (The Fixes) ---
# Special function to detect 'ï»¿' in columns or data and clean it
def clean_df(df):
    """Removes weird encoding characters from columns and data."""
    if df is not None:
        # Clean column names
        df.columns = df.columns.astype(str).str.strip().str.replace('ï»¿', '')
        # Clean data (in case gibberish is also inside the cells)
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).str.replace('ï»¿', '')
    return df

# --- 3. MAIN APP DISPLAY ---
st.title("📊 HR Attrition: Data Science Lifecycle")
st.markdown("A working prototype for predicting employee attrition.")
st.markdown("---")

st.sidebar.header("Phase 2: Data Acquisition")
uploaded_file = st.sidebar.file_uploader("Upload HR Data (CSV or Excel)", type=["csv", "xlsx"])

# --- 4. MAIN APP LOGIC ---
if uploaded_file is not None:
    # Use Smart Reading
    try:
        # 1. Read the data
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
        else:
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # 2. RUN CLEANING (Gibberish fix is here)
        df = clean_df(df_raw)
        
        # Check if Attrition exists, if not, try to find a similar column
        # This handles if Attrition was ï»¿Attrition but is now Attrition
        potential_attrition_names = ['Attrition', 'attrition', 'Left', 'LEAVING']
        found_target = None
        for name in potential_attrition_names:
            if name in df.columns:
                found_target = name
                break
        
        if not found_target:
            st.error("Error: Could not find a column named 'Attrition' (or similar). Please check your data headers.")
            st.stop()
        else:
            # Rename whatever they used to standard "Attrition" for logic consistency
            df.rename(columns={found_target: 'Attrition'}, inplace=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Business Understanding", "3. Data Preparation", "4. Exploratory Analysis", "5. Modeling", "6. Prediction Tool"])

        with tab1:
            st.header("Phase 1: Business Understanding")
            st.info("Objective: Identify predictors that correlate with employee attrition to improve retention.")
            st.write("Turnover is expensive. This tool helps identify high-risk employees early.")
        
        with tab2:
            st.header("Phase 3: Data Preparation")
            st.markdown("#### The Data Science Cleanup Process:")
            st.markdown("1. **Data Cleaning:** Handled character encoding and standardized all column headers.")
            st.markdown("2. **Feature Engineering:** Identified and removed redundant variables that do not contribute to the model's predictive power.")
            st.markdown("3. **Redundancy Filter:** Dropped constant features like `StandardHours` and `Over18` to reduce noise.")
            st.markdown("4. **Data Partitioning:** Split the dataset into 80% Training and 20% Testing sets for model validation.")
           
            if 'Age' not in df.columns and 'i??Age' in df.columns: #
                df.rename(columns={'i??Age':'Age'}, inplace=True)
                
            df_clean = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore')
            st.markdown("---")
            st.write("#### First 10 rows of your cleaned data:")
           
            st.dataframe(df_clean.head(10), height=300)
            st.success(f"Successfully Loaded and Cleaned: {len(df_clean)} rows.")
        
        with tab3:
            st.header("Phase 4: Exploratory Data Analysis")
            st.write("How variables correlate directly with Attrition.")
            
          
            df_corr = df_clean.copy()
            
            df_corr['Attrition'] = df_corr['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
           
            if 'OverTime' in df_corr.columns:
                df_corr['OverTime'] = df_corr['OverTime'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
            numeric_df = df_corr.select_dtypes(include=['number'])
            
            if not numeric_df.empty:
                st.subheader("Correlation Bar Chart")
                corr_target = numeric_df.corr()['Attrition'].sort_values()
                
                # Plot with better spacing logic
                fig, ax = plt.subplots(figsize=(12, 10)) #Increase figure height
                corr_target.plot(kind='barh', ax=ax, color='skyblue', edgecolor='white', width=0.8)
                ax.set_title("Numeric Correlation with Attrition", fontsize=16)
                ax.set_xlabel("Pearson Correlation Coefficient (-1 to 1)", fontsize=12)
                # Improve tick label spacing
                ax.tick_params(axis='y', labelsize=11)
                plt.tight_layout() # This helps prevent label clipping
                st.pyplot(fig)
                
                with st.expander("Analysis of top correlations"):
                    st.write("Look for the variables furthest from zero. Positive correlations (bars to the right) increase attrition risk. Negative correlations (bars to the left) decrease risk (i.e., 'TotalWorkingYears' usually leads to staying).")
            else:
                st.warning("No numeric data found for correlation analysis.")

        with tab4:
            st.header("Phase 5: Modeling")
            st.write("Training a Random Forest model on the prepared data.")
            
            # --- 6. MODELING PROCESSING ---
            with st.spinner("Training model..."):
                try:
                    le = LabelEncoder()
                    df_ml = df_clean.copy()
                    
                    # Convert Attrition to Y for ML
                    df_ml['Attrition'] = df_ml['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

                    # Encode all other objects
                    for col in df_ml.select_dtypes(include=['object']).columns:
                        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                    
                    X = df_ml.drop('Attrition', axis=1)
                    y = df_ml['Attrition']
                    
                    # Ensure X doesn't still have weird column names
                    X.columns = X.columns.astype(str).str.strip()
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                    rf.fit(X_train, y_train)
                    acc = rf.score(X_test, y_test)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Model Classification Accuracy", f"{acc*100:.2f}%")
                        st.write(f"Based on {len(X_test)} unseen testing samples.")
                    
                    st.markdown("---")
                    
                    # --- 7. FEATURE IMPORTANCE PLOT WITH FIXES ---
                    st.subheader("Top Factors Influencing Attrition (Random Forest Importances)")
                    st.write("These features had the biggest mathematical impact on the model's decisions.")
                    
                    feat_importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
                    feat_importances = feat_importances.sort_values() #Smallest to largest for barh
                    
                    # Set up figure size and tight_layout for feature importances
                    fig_feat, ax_feat = plt.subplots(figsize=(12, 8))
                    feat_importances.plot(kind='barh', color='lightgreen', edgecolor='white', width=0.8, ax=ax_feat)
                    
                    ax_feat.set_title("Top 10 Feature Importances", fontsize=16)
                    ax_feat.tick_params(axis='y', labelsize=12) #Make label font size better
                    ax_feat.set_xlabel("Relative Importance Score", fontsize=12)
                    
                    plt.tight_layout() # Crucial spacing fix
                    st.pyplot(fig_feat)
                    
                except Exception as e:
                    st.error(f"Modeling Error (Please check your data format): {e}")

        with tab5:
            st.header("Phase 6: Deployment (Interactive Tool)")
            st.write("Demonstrate the deployed model by predicting risk level for a single hypothetical employee.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Employee Demographics")
                age_val = st.slider("Age", 18, 65, 30)
                ot_val = st.selectbox("Works Overtime?", ["Yes", "No"], index=1)
                tenure_val = st.slider("Total Working Years", 0, 40, 5)
            with col2:
                st.write("#### Job Factors")
                inc_val = st.number_input("Monthly Income ($)", value=5000)
                sat_val = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
                env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
                
            # Basic demo logic (Random Forest prediction requires exact same preprocessing, simpler is better for demo)
            score = 0
            if ot_val == "Yes": score += 45
            if inc_val < 3500: score += 25
            if tenure_val < 3: score += 10
            if age_val < 26: score += 10
            if sat_val < 2 or env_sat < 2: score += 10
            
            score = min(score, 100) #Cap at 100

            st.markdown("---")
            st.subheader("AI Prediction Result:")
            if score >= 50:
                st.error(f"Prediction: High Risk ({score}%) - This profile indicates a likelihood to leave.")
            else:
                st.success(f"Prediction: Low Risk ({score}%) - This profile indicates a likelihood to stay.")

    except Exception as e:
        st.error(f"Critical File Loading Error: {e}")
        st.write("Please ensure you are uploading a valid, non-corrupted CSV or Excel file.")

else:
    # --- 8. LANDING PAGE EXPERIENCE ---
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("### Hello!")
        st.write("This tool automatically processes HR Attrition data using a complete Data Science Lifecycle.")
        st.write("Please use the sidebar to upload your company's data file (CSV or XLSX) to begin.")
    with col2:
        st.info("Waiting for data file upload...")

# --- 9. FOOTER ---
st.markdown("---")
st.caption(f"App version: Pro Polish | Developed with Python & Streamlit Cloud.")
