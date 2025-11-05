import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_models_and_preprocessors():
    """
    Load trained models and preprocessing objects.
    """
    try:
        iso_model = joblib.load('demo/threat_hunting_ml/isolation_forest_model.pkl')
        rf_model = joblib.load('demo/threat_hunting_ml/random_forest_model.pkl')
        scaler = joblib.load('demo/threat_hunting_ml/scaler.pkl')
        le_event = joblib.load('demo/threat_hunting_ml/le_event.pkl')
        le_user = joblib.load('demo/threat_hunting_ml/le_user.pkl')
        le_ip = joblib.load('demo/threat_hunting_ml/le_ip.pkl')
        features = joblib.load('demo/threat_hunting_ml/features.pkl')
        return iso_model, rf_model, scaler, le_event, le_user, le_ip, features
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run model.py first to train and save models.")
        return None, None, None, None, None, None, None

def detect_threats(log_data, threshold=0.7):
    """
    Detect threats in security logs using trained ML models.

    Parameters:
    - log_data: DataFrame with security log entries
    - threshold: Probability threshold for threat classification (0-1)

    Returns:
    - DataFrame with threat scores and predictions
    """
    models_loaded = load_models_and_preprocessors()
    if not all(models_loaded):
        return None

    iso_model, rf_model, scaler, le_event, le_user, le_ip, features = models_loaded

    # Preprocess the data
    from preprocess import preprocess_new_data
    X_processed = preprocess_new_data(log_data.copy(), scaler, le_event, le_user, le_ip, features)

    # Get predictions from both models
    iso_predictions = iso_model.predict(X_processed)
    iso_scores = iso_model.decision_function(X_processed)
    rf_predictions = rf_model.predict(X_processed)
    rf_probabilities = rf_model.predict_proba(X_processed)[:, 1]

    # Convert Isolation Forest predictions (-1, 1) to (1, 0) for anomalies
    iso_anomaly_scores = np.where(iso_predictions == -1, 1, 0)

    # Combine predictions (ensemble approach)
    combined_score = (iso_anomaly_scores + rf_probabilities) / 2
    threat_prediction = (combined_score > threshold).astype(int)

    # Add results to original data
    results_df = log_data.copy()
    results_df['iso_anomaly_score'] = iso_anomaly_scores
    results_df['rf_threat_probability'] = rf_probabilities
    results_df['combined_threat_score'] = combined_score
    results_df['is_threat'] = threat_prediction
    results_df['threat_level'] = pd.cut(combined_score,
                                       bins=[0, 0.3, 0.7, 1.0],
                                       labels=['Low', 'Medium', 'High'])

    return results_df

def generate_sample_logs(num_logs=100):
    """
    Generate sample logs for testing threat detection.
    """
    from data_generator import generate_security_logs
    return generate_security_logs(num_logs)

def analyze_threats(results_df):
    """
    Analyze detected threats and provide summary statistics.
    """
    if results_df is None or results_df.empty:
        return "No data to analyze."

    total_logs = len(results_df)
    threats_detected = results_df['is_threat'].sum()
    threat_percentage = (threats_detected / total_logs) * 100

    threat_by_type = results_df[results_df['is_threat'] == 1]['event_type'].value_counts()
    threat_by_user = results_df[results_df['is_threat'] == 1]['user_id'].value_counts().head(10)
    threat_by_hour = results_df[results_df['is_threat'] == 1]['hour_of_day'].value_counts().sort_index()

    analysis = f"""
Threat Detection Analysis:
========================
Total Logs Analyzed: {total_logs}
Threats Detected: {threats_detected} ({threat_percentage:.2f}%)

Threats by Event Type:
{threat_by_type.to_string()}

Top Users with Threats:
{threat_by_user.to_string()}

Threats by Hour of Day:
{threat_by_hour.to_string()}
"""

    return analysis

if __name__ == "__main__":
    print("Generating sample security logs for testing...")
    sample_logs = generate_sample_logs(500)

    print("Detecting threats...")
    results = detect_threats(sample_logs)

    if results is not None:
        print("Threat detection completed!")
        print(analyze_threats(results))

        # Save results
        results.to_csv('demo/threat_hunting_ml/threat_detection_results.csv', index=False)
        print("Results saved to threat_detection_results.csv")
    else:
        print("Failed to load models. Please train models first by running model.py")
