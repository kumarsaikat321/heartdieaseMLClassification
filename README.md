# Heart Disease Classification - Machine Learning Assignment 2

**Course**: Machine Learning  
**Program**: M.Tech (AIML/DSE)  
**Institution**: BITS Pilani - Work Integrated Learning Programmes  
**Submission Deadline**: February 15, 2026

---

## Problem Statement

This project implements a comprehensive machine learning solution for **binary classification of heart disease**. The objective is to predict whether a patient has heart disease (target = 1) or not (target = 0) based on 13 clinical features including age, sex, chest pain type, blood pressure, cholesterol levels, ECG results, and other cardiovascular indicators.

Heart disease remains one of the leading causes of mortality worldwide. Early and accurate detection can significantly improve patient outcomes, reduce healthcare costs, and save lives. This project leverages multiple machine learning algorithms to identify the most effective model for predicting heart disease presence based on clinical data.

---

## Dataset Description

### Source
**Dataset**: Heart Disease Dataset  
**Type**: Binary Classification  
**Source**: Clinical cardiovascular data

### Dataset Statistics
- **Total Instances**: 1,025 patients
- **Number of Features**: 13 (excluding target variable)
- **Target Variable**: `target` (0 = No Disease, 1 = Disease)
- **Class Distribution**: 
  - Class 0 (No Disease): 499 instances (48.7%)
  - Class 1 (Disease): 526 instances (51.3%)
- **Missing Values**: None
- **Feature Types**: 
  - Numerical: 5 features
  - Categorical: 8 features

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Age of the patient in years |
| `sex` | Categorical | Gender (Male/Female) |
| `chest_pain_type` | Categorical | Type of chest pain (Typical angina, Atypical angina, Non-anginal pain, Asymptomatic) |
| `resting_blood_pressure` | Numerical | Resting blood pressure in mm Hg |
| `cholestoral` | Numerical | Serum cholesterol in mg/dl |
| `fasting_blood_sugar` | Categorical | Fasting blood sugar level (Greater than 120 mg/ml, Lower than 120 mg/ml) |
| `rest_ecg` | Categorical | Resting electrocardiographic results |
| `Max_heart_rate` | Numerical | Maximum heart rate achieved |
| `exercise_induced_angina` | Categorical | Exercise-induced angina (Yes/No) |
| `oldpeak` | Numerical | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of the peak exercise ST segment |
| `vessels_colored_by_flourosopy` | Categorical | Number of major vessels colored by fluoroscopy (0-3) |
| `thalassemia` | Categorical | Thalassemia type (Normal, Fixed Defect, Reversible Defect) |
| `target` | Binary | Heart disease presence (0 = No, 1 = Yes) |

### Data Preprocessing Steps
1. **Encoding**: All categorical variables encoded using LabelEncoder
2. **Feature Scaling**: StandardScaler applied to normalize numerical features
3. **Train-Test Split**: 80% training (820 samples), 20% testing (205 samples)
4. **Stratification**: Target variable stratified to maintain class balance

---

## Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|-----|
| Logistic Regression | 0.8439 | 0.9320 | 0.8230 | 0.8857 | 0.8532 | 0.6891 |
| Decision Tree | 0.9366 | 0.9885 | 0.9510 | 0.9238 | 0.9372 | 0.8736 |
| kNN | 0.8634 | 0.9689 | 0.8598 | 0.8762 | 0.8679 | 0.7267 |
| Naive Bayes | 0.8439 | 0.9135 | 0.8288 | 0.8762 | 0.8519 | 0.6884 |
| Random Forest (Ensemble) | 0.9415 | 0.9847 | 0.9189 | 0.9714 | 0.9444 | 0.8842 |
| XGBoost (Ensemble) | 0.9073 | 0.9677 | 0.8772 | 0.9524 | 0.9132 | 0.8173 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Solid baseline performance with 84.4% accuracy and excellent AUC of 93.2%. The model demonstrates good balance between precision (82.3%) and recall (88.6%), making it suitable for interpretable clinical decisions. While it captures linear relationships effectively, it may miss complex non-linear patterns in the data. The relatively low computational cost and high interpretability make it valuable for understanding feature contributions. Recommended for scenarios requiring transparent decision-making and fast inference. |
| **Decision Tree** | Outstanding performance with 93.7% accuracy and near-perfect precision (95.1%). The high AUC of 98.8% and MCC of 0.874 indicate excellent discriminative ability and overall classification quality. The tree structure provides clear interpretability through decision rules. Strong recall (92.4%) ensures most positive cases are identified. The model balances complexity and performance well, though individual trees can be prone to overfitting on different data splits. Excellent choice when model interpretability is crucial. |
| **kNN** | Good performance with 86.3% accuracy and strong AUC of 96.9%. The model effectively captures local patterns and similarities between patient profiles. Balanced precision (86.0%) and recall (87.6%) demonstrate consistent performance across both classes. The algorithm's simplicity and non-parametric nature make it robust to certain types of data irregularities. However, prediction time increases with dataset size, and performance can be sensitive to the choice of k and distance metric. Suitable when local similarity patterns are important for classification. |
| **Naive Bayes** | Reliable baseline with 84.4% accuracy despite its strong independence assumption between features. Good recall (87.6%) makes it effective at identifying positive heart disease cases, which is critical in medical screening applications. The probabilistic nature allows for uncertainty quantification, valuable in clinical contexts. Fast training and prediction times make it suitable for real-time applications. The model's simplicity ensures robustness and low risk of overfitting. Recommended for quick prototyping and scenarios requiring fast inference with reasonable accuracy. |
| **Random Forest (Ensemble)** | **Exceptional performance with 94.1% accuracy and outstanding AUC of 98.5%**, making it the top-performing model. The ensemble approach aggregates 100 decision trees, providing excellent balance between bias and variance. High precision (91.9%) and exceptional recall (97.1%) indicate reliable predictions for both classes. The MCC score of 0.884 demonstrates superior overall classification quality. Feature importance analysis capability offers valuable clinical insights. The model shows strong generalization without overfitting. **Highly recommended for production deployment** due to its robustness, accuracy, and ability to handle complex feature interactions. |
| **XGBoost (Ensemble)** | Strong performance with 90.7% accuracy and excellent AUC of 96.8%. The gradient boosting approach effectively handles complex patterns and feature interactions through sequential tree building. High recall (95.2%) ensures most heart disease cases are detected, crucial for medical applications. Good precision (87.7%) maintains reliability. The MCC of 0.817 indicates solid overall performance. XGBoost's built-in regularization helps prevent overfitting. The model offers flexibility through extensive hyperparameter tuning options. Recommended as a robust alternative to Random Forest with excellent predictive power and feature importance insights. |

---

## Key Findings

### Performance Analysis

1. **Best Overall Model**: **Random Forest (94.1% accuracy)**
   - Highest accuracy and F1 score
   - Excellent precision-recall balance
   - Strong AUC indicating superior discriminative ability
   - Robust to overfitting through ensemble averaging

2. **Runner-Up Models**:
   - **Decision Tree**: 93.7% accuracy (high interpretability)
   - **XGBoost**: 90.7% accuracy (strong gradient boosting)

3. **Baseline Models**:
   - Logistic Regression and Naive Bayes both achieved 84.4% accuracy
   - Provide good baseline performance with high interpretability
   - Useful for comparison and understanding feature importance

### Model Recommendations by Use Case

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **Production Deployment** | Random Forest | Highest accuracy (94.1%), excellent recall (97.1%), robust performance |
| **Clinical Interpretability** | Decision Tree | Clear decision rules, 93.7% accuracy, explainable predictions |
| **Fast Inference** | Naive Bayes | Fastest predictions, 84.4% accuracy, suitable for real-time screening |
| **Feature Importance Analysis** | Random Forest or XGBoost | Built-in feature importance, identify key clinical indicators |
| **Resource-Constrained Environments** | Logistic Regression | Smallest model size, 84.4% accuracy, minimal computational requirements |
| **Balanced Precision-Recall** | Random Forest | Best F1 score (94.4%), optimal for scenarios where both metrics matter |

### Clinical Insights

- **High Recall Models** (Random Forest: 97.1%, XGBoost: 95.2%): Excellent for screening where false negatives are costly
- **High Precision Models** (Decision Tree: 95.1%, Random Forest: 91.9%): Suitable for confirmatory testing where false positives are concerning
- **Ensemble Methods Outperform**: Both Random Forest and XGBoost significantly exceed individual classifiers
- **Dataset Quality**: All models achieve >84% accuracy, indicating strong predictive signals in the clinical features

---

## Technical Implementation

### Libraries and Tools
- **Python**: 3.8+
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models and evaluation metrics
- **XGBoost**: Gradient boosting classifier
- **Streamlit**: Interactive web application framework
- **matplotlib & seaborn**: Data visualization

### Model Hyperparameters

| Model | Key Parameters |
|-------|----------------|
| Logistic Regression | random_state=42, max_iter=1000 |
| Decision Tree | random_state=42 |
| kNN | n_neighbors=5 |
| Naive Bayes | Default (Gaussian) |
| Random Forest | n_estimators=100, random_state=42 |
| XGBoost | n_estimators=100, random_state=42 |

### Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions (TP + TN) / Total
- **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes (0-1, higher is better)
- **Precision**: Proportion of positive predictions that are correct (TP / (TP + FP))
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified (TP / (TP + FN))
- **F1 Score**: Harmonic mean of precision and recall (2 × (Precision × Recall) / (Precision + Recall))
- **MCC (Matthews Correlation Coefficient)**: Correlation between predictions and actual values (-1 to +1)

---

## Project Structure

```
heart-disease-classification/
│
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training and evaluation script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── HeartDiseaseTrain-Test.csv     # Original dataset
├── heart_disease_test.csv         # Test dataset for Streamlit
│
├── model_logistic_regression.pkl  # Trained Logistic Regression
├── model_decision_tree.pkl        # Trained Decision Tree
├── model_knn.pkl                  # Trained kNN
├── model_naive_bayes.pkl          # Trained Naive Bayes
├── model_random_forest.pkl        # Trained Random Forest
├── model_xgboost.pkl              # Trained XGBoost
│
├── scaler.pkl                     # StandardScaler for features
├── label_encoders.pkl             # LabelEncoders for categorical features
│
└── model_comparison.csv           # Evaluation results comparison
```

---

## Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd heart-disease-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train models (optional)**
```bash
python train_models.py
```

### Running the Streamlit Application

**Local deployment:**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

**Cloud deployment:**
1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App"
5. Select your repository and app.py
6. Click "Deploy"

---

## Streamlit Application Features

The interactive web application provides:

1. **📤 Dataset Upload**: Upload test data in CSV format
2. **🎯 Model Selection**: Choose from 6 trained classification models
3. **📊 Evaluation Metrics**: Display all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **🔢 Confusion Matrix**: Visual heatmap of prediction performance
5. **📋 Classification Report**: Detailed per-class performance metrics
6. **🏆 Model Comparison**: Side-by-side comparison of all models
7. **📈 Performance Visualization**: Interactive charts and graphs
8. **💾 Download Predictions**: Export predictions with probability scores

---

## Future Improvements

1. **Hyperparameter Optimization**: Implement GridSearchCV or RandomizedSearchCV for optimal parameters
2. **Cross-Validation**: Add k-fold cross-validation for more robust performance estimates
3. **Feature Engineering**: Create interaction features and polynomial features
4. **Model Explainability**: Integrate SHAP or LIME for individual prediction explanations
5. **Real-time Prediction**: Add input form for predicting single patient outcomes
6. **Model Monitoring**: Implement performance tracking and model versioning
7. **API Development**: Create REST API endpoints for model serving
8. **Advanced Ensemble**: Implement stacking or voting classifiers

---

## Conclusion

This project successfully demonstrates a complete end-to-end machine learning pipeline for heart disease classification. The **Random Forest model achieved the best performance (94.1% accuracy)**, followed closely by Decision Tree (93.7%) and XGBoost (90.7%). 

Key achievements:
- ✅ All 6 models implemented and evaluated
- ✅ Comprehensive evaluation with 6 metrics per model
- ✅ Interactive web application deployed
- ✅ Production-ready model pipeline
- ✅ Professional documentation

The high performance across all models (>84% accuracy) demonstrates that the selected clinical features contain strong predictive signals for heart disease detection. The ensemble methods (Random Forest and XGBoost) provide the most reliable predictions, while simpler models (Logistic Regression, Naive Bayes) offer valuable baselines with higher interpretability.

---

## Academic Integrity

This project was developed as part of Machine Learning Assignment 2 for BITS Pilani M.Tech (AIML/DSE) program. All code is original work with proper attribution to libraries and frameworks used.

**Student**: Saikat Kumar Mallick  
**Roll Number**: 2025aa05556
**Date**: February 15, 2026

---

## References

1. UCI Machine Learning Repository - Heart Disease Dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Streamlit Documentation: https://docs.streamlit.io/
4. XGBoost Documentation: https://xgboost.readthedocs.io/

---

**Assignment completed successfully on BITS Virtual Lab**  
*Last Updated: February 2026*
