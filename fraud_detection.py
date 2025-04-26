import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
from sklearn.preprocessing import LabelEncoder

# 1. Load Dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# 2. Data Preprocessing
def preprocess_data(data):
    # Handle missing values
    data = data.ffill()

    # Label encode high-cardinality categorical features
    label_enc = LabelEncoder()
    data['trans_date_trans_time'] = label_enc.fit_transform(data['trans_date_trans_time'])
    data['merchant'] = label_enc.fit_transform(data['merchant'])

    # Normalize numerical features
    data['amt'] = (data['amt'] - data['amt'].mean()) / data['amt'].std()

    # Extract time-based features
    data['transaction_hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['transaction_day'] = pd.to_datetime(data['trans_date_trans_time']).dt.dayofweek

    # Drop unnecessary columns
    data = data.drop(['trans_date_trans_time'], axis=1)

    return data

# 3. Feature Selection
def select_features(data):
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    return X, y

# 4. Handle Imbalanced Dataset
def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# 5. Build and Train Model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# 6. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# 6.1 Optimizing memory 
def optimize_memory(data):
    for col in data.select_dtypes(include=['int', 'float']):
        data[col] = pd.to_numeric(data[col], downcast='float')
    return data

# 7. Explainability
def explain_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test)

# Main Execution Flow
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data("fraudTest.csv")
    data = preprocess_data(data)
    X, y = select_features(data)
    X_resampled, y_resampled = balance_data(X, y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    # Explain model predictions
    explain_model(model, X_test)