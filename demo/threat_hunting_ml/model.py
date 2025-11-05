import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_isolation_forest(X_train, X_test, y_test):
    """
    Train Isolation Forest for anomaly detection.
    """
    print("Training Isolation Forest...")

    # Use contamination based on known anomaly ratio
    anomaly_ratio = np.mean(y_test)
    contamination = max(0.05, min(0.5, anomaly_ratio * 1.5))  # Adaptive contamination

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    iso_forest.fit(X_train)

    print(f"Using contamination: {contamination}")

    # Predictions
    y_pred_train = iso_forest.predict(X_train)
    y_pred_test = iso_forest.predict(X_test)

    # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
    y_pred_test_binary = np.where(y_pred_test == -1, 1, 0)

    print("Isolation Forest Results:")
    print(classification_report(y_test, y_pred_test_binary))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_test_binary):.3f}")

    return iso_forest

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest Classifier for supervised threat detection.
    """
    print("Training Random Forest Classifier...")

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'random_state': [42]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("Random Forest Results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")

    return best_model

def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'demo/threat_hunting_ml/{title.lower().replace(" ", "_")}.png')
    plt.close()

def save_model(model, filename):
    """
    Save trained model to disk.
    """
    joblib.dump(model, f'demo/threat_hunting_ml/{filename}')
    print(f"Model saved as {filename}")

def load_model(filename):
    """
    Load trained model from disk.
    """
    return joblib.load(f'demo/threat_hunting_ml/{filename}')

if __name__ == "__main__":
    from preprocess import load_and_preprocess_data

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, le_event, le_user, le_ip, features = load_and_preprocess_data('demo/threat_hunting_ml/security_logs.csv')

    # Train Isolation Forest
    iso_model = train_isolation_forest(X_train, X_test, y_test)
    save_model(iso_model, 'isolation_forest_model.pkl')

    # Train Random Forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)
    save_model(rf_model, 'random_forest_model.pkl')

    # Save preprocessing objects
    joblib.dump(scaler, 'demo/threat_hunting_ml/scaler.pkl')
    joblib.dump(le_event, 'demo/threat_hunting_ml/le_event.pkl')
    joblib.dump(le_user, 'demo/threat_hunting_ml/le_user.pkl')
    joblib.dump(le_ip, 'demo/threat_hunting_ml/le_ip.pkl')
    joblib.dump(features, 'demo/threat_hunting_ml/features.pkl')

    print("Models trained and saved successfully!")
