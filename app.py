"""
Heart Disease Prediction - Streamlit Application
Interactive web application for heart disease classification using 6 ML models

Author: Saikat Kumar Mallick
Course: Machine Learning - BITS Pilani M.Tech (AIML/DSE)
Date: February 15, 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>

/* Main padding */
.main {padding: 0rem 1rem;}

/* Metric card background */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e3a8a, #0f172a); !important;  /* Dark Blue */
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 12px;
}

/* Metric label */
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    color: #22c55e !important;  /* Green numbers */
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE
# ============================================================================

st.title("❤️ Heart Disease Prediction System")
st.markdown("---")
st.markdown("""
This interactive application demonstrates **6 Machine Learning classification models** 
for predicting heart disease based on clinical features.
""")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("📊 Configuration")
st.sidebar.markdown("### Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])

MODEL_FILES = {
    'Logistic Regression': 'model_logistic_regression.pkl',
    'Decision Tree': 'model_decision_tree.pkl',
    'KNN': 'model_knn.pkl',
    'Naive Bayes': 'model_naive_bayes.pkl',
    'Random Forest': 'model_random_forest.pkl',
    'XGBoost': 'model_xgboost.pkl'
}

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    models = {}

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(BASE_DIR, filename)

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {filename}")
            st.write("Current directory files:", os.listdir(BASE_DIR))
            return None

        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)

    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

    if not os.path.exists(scaler_path):
        st.error("Scaler file not found!")
        st.write("Current directory files:", os.listdir(BASE_DIR))
        return None

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return models, scaler

# ============================================================================
# MAIN APP
# ============================================================================

if uploaded_file is not None:
    try:
        # Load data
        test_data = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset uploaded!")
        st.sidebar.metric("Samples", len(test_data))
        
        # Validate
        if 'target' not in test_data.columns:
            st.error("❌ Dataset must contain a 'target' column!")
            st.stop()
        
        # Load models
        models_data = load_models()
        if models_data is None:
            st.stop()
        
        models, scaler = models_data
        
        # Prepare data
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Select Model")
        selected_model = st.sidebar.selectbox(
            "Choose a model:",
            list(MODEL_FILES.keys())
        )
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"📈 Model: {selected_model}")
        with col2:
            st.metric("Test Samples", len(y_test))
        
        st.markdown("---")
        
        # Get model and predict
        model = models[selected_model]
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display metrics
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("AUC Score", f"{auc:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix and Report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred,
                                          target_names=['No Disease', 'Disease'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), 
                        use_container_width=True)
        
        st.markdown("---")
        
        # Model Comparison
        st.subheader("🏆 All Models Comparison")
        
        all_results = []
        for model_name, model_obj in models.items():
            y_pred_temp = model_obj.predict(X_test_scaled)
            y_pred_proba_temp = model_obj.predict_proba(X_test_scaled)[:, 1]
            
            all_results.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred_temp),
                'AUC': roc_auc_score(y_test, y_pred_proba_temp),
                'Precision': precision_score(y_test, y_pred_temp, average='binary'),
                'Recall': recall_score(y_test, y_pred_temp, average='binary'),
                'F1': f1_score(y_test, y_pred_temp, average='binary'),
                'MCC': matthews_corrcoef(y_test, y_pred_temp)
            })
        
        comparison_df = pd.DataFrame(all_results)
        
        st.dataframe(
            comparison_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            }).background_gradient(subset=['Accuracy', 'AUC', 'F1'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Performance Visualization
        st.markdown("---")
        st.subheader("📊 Performance Visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_list)):
            data = comparison_df.sort_values(metric, ascending=True)
            colors = ['crimson' if m == selected_model else 'steelblue' 
                     for m in data['Model']]
            ax.barh(data['Model'], data[metric], color=colors)
            ax.set_xlabel(metric)
            ax.set_xlim(0, 1)
            ax.set_title(f'{metric} Comparison')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Download predictions
        st.markdown("---")
        st.subheader("💾 Download Predictions")
        
        predictions_df = test_data.copy()
        predictions_df['predicted'] = y_pred
        predictions_df['probability'] = y_pred_proba
        
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"predictions_{selected_model.replace(' ', '_')}.csv",
            "text/csv"
        )
        
        # Dataset preview
        with st.expander("📄 View Dataset Preview"):
            st.dataframe(test_data.head(20), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👈 Please upload a test dataset (CSV) to begin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 About")
        st.markdown("""
        **Dataset**: Heart Disease (1,025 patients)  
        **Features**: 13 clinical attributes  
        **Target**: Binary (0=No Disease, 1=Disease)
        
        **How to Use:**
        1. Upload CSV from sidebar
        2. Select a model
        3. View metrics and predictions
        """)
    
    with col2:
        st.subheader("🤖 Models")
        st.markdown("""
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors
        4. Naive Bayes
        5. Random Forest 🏆
        6. XGBoost
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Heart Disease Prediction | ML Assignment 2 | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
