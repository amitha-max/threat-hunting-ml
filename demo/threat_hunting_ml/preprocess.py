import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path='security_logs.csv'):
    """
    Load and preprocess security log data for ML modeling.
    """
    # Load data
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Feature engineering
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Encode categorical variables
    le_event = LabelEncoder()
    le_user = LabelEncoder()
    le_ip = LabelEncoder()

    df['event_type_encoded'] = le_event.fit_transform(df['event_type'])
    df['user_id_encoded'] = le_user.fit_transform(df['user_id'])
    df['ip_address_encoded'] = le_ip.fit_transform(df['ip_address'])

    # Select features for modeling
    features = [
        'event_type_encoded', 'user_id_encoded', 'ip_address_encoded',
        'login_attempts', 'session_duration', 'data_transferred',
        'hour_of_day', 'day_of_week', 'month', 'is_weekend'
    ]

    X = df[features]
    y = df['is_anomaly']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, le_event, le_user, le_ip, features

def preprocess_new_data(new_data, scaler, le_event, le_user, le_ip, features):
    """
    Preprocess new incoming data for prediction.
    """
    # Feature engineering for new data
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
    new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek
    new_data['month'] = new_data['timestamp'].dt.month
    new_data['is_weekend'] = new_data['day_of_week'].isin([5, 6]).astype(int)

    # Encode categorical variables
    new_data['event_type_encoded'] = le_event.transform(new_data['event_type'])
    new_data['user_id_encoded'] = le_user.transform(new_data['user_id'])
    new_data['ip_address_encoded'] = le_ip.transform(new_data['ip_address'])

    X_new = new_data[features]
    X_new_scaled = scaler.transform(X_new)

    return X_new_scaled

if __name__ == "__main__":
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, le_event, le_user, le_ip, features = load_and_preprocess_data('demo/threat_hunting_ml/security_logs.csv')
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Anomaly ratio in training: {y_train.mean():.3f}")
