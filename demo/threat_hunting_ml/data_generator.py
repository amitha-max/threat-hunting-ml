import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_security_logs(num_logs=10000):
    """
    Generate simulated security event logs for threat hunting.
    """
    np.random.seed(42)
    random.seed(42)

    # Base timestamp
    base_time = datetime.now() - timedelta(days=30)

    # Event types
    event_types = ['LOGIN', 'LOGOUT', 'FILE_ACCESS', 'NETWORK_CONNECTION', 'PROCESS_START', 'REGISTRY_CHANGE']

    # User IDs
    user_ids = [f'user_{i}' for i in range(1, 101)]

    # IP addresses
    ip_addresses = [f'192.168.1.{i}' for i in range(1, 256)]

    # File paths
    file_paths = ['/home/user/documents/', '/var/log/', '/etc/', '/tmp/', '/usr/bin/']

    # Process names
    process_names = ['ssh', 'apache2', 'mysql', 'python', 'bash', 'firefox', 'chrome']

    logs = []

    for i in range(num_logs):
        timestamp = base_time + timedelta(minutes=random.randint(0, 43200))  # 30 days in minutes

        event_type = random.choice(event_types)
        user_id = random.choice(user_ids)
        ip = random.choice(ip_addresses)

        # Simulate some anomalous behavior
        is_anomaly = random.random() < 0.05  # 5% anomalies

        if is_anomaly:
            # Anomalous patterns
            if random.random() < 0.5:
                # Unusual login times
                hour = random.choice([2, 3, 4, 5])  # Late night hours
            else:
                # Unusual file access
                file_path = '/etc/passwd' if random.random() < 0.3 else '/root/.ssh/'
                event_type = 'FILE_ACCESS'
        else:
            hour = random.randint(8, 18)  # Normal business hours
            file_path = random.choice(file_paths)

        process_name = random.choice(process_names)

        # Generate some features
        login_attempts = random.randint(1, 5) if event_type == 'LOGIN' else 0
        session_duration = random.randint(10, 3600) if event_type in ['LOGIN', 'LOGOUT'] else 0
        data_transferred = random.randint(100, 1000000) if event_type == 'NETWORK_CONNECTION' else 0

        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip,
            'file_path': file_path if event_type == 'FILE_ACCESS' else '',
            'process_name': process_name,
            'login_attempts': login_attempts,
            'session_duration': session_duration,
            'data_transferred': data_transferred,
            'hour_of_day': hour,
            'is_anomaly': int(is_anomaly)
        }

        logs.append(log_entry)

    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

if __name__ == "__main__":
    print("Generating security logs...")
    logs_df = generate_security_logs(10000)
    logs_df.to_csv('demo/threat_hunting_ml/security_logs.csv', index=False)
    print(f"Generated {len(logs_df)} security log entries.")
    print(f"Anomalies: {logs_df['is_anomaly'].sum()}")
