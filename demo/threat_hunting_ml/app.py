from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from detect import detect_threats, analyze_threats, generate_sample_logs
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    try:
        num_logs = int(request.form.get('num_logs', 1000))
        logs_df = generate_sample_logs(num_logs)
        logs_df.to_csv('demo/threat_hunting_ml/security_logs.csv', index=False)
        return jsonify({'status': 'success', 'message': f'Generated {num_logs} security logs'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/detect_threats', methods=['POST'])
def detect():
    try:
        if not os.path.exists('demo/threat_hunting_ml/security_logs.csv'):
            return jsonify({'status': 'error', 'message': 'No security logs found. Generate data first.'})

        logs_df = pd.read_csv('demo/threat_hunting_ml/security_logs.csv')
        results = detect_threats(logs_df)

        if results is None:
            return jsonify({'status': 'error', 'message': 'Models not trained. Train models first.'})

        results.to_csv('demo/threat_hunting_ml/threat_detection_results.csv', index=False)

        # Generate summary statistics
        analysis = analyze_threats(results)
        threat_summary = {
            'total_logs': len(results),
            'threats_detected': int(results['is_threat'].sum()),
            'threat_percentage': round((results['is_threat'].sum() / len(results)) * 100, 2)
        }

        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'summary': threat_summary
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        from model import train_isolation_forest, train_random_forest, save_model
        from preprocess import load_and_preprocess_data
        import joblib

        print("Loading data...")
        X_train, X_test, y_train, y_test, scaler, le_event, le_user, le_ip, features = load_and_preprocess_data('demo/threat_hunting_ml/security_logs.csv')

        print("Training models...")
        iso_model = train_isolation_forest(X_train, X_test, y_test)
        rf_model = train_random_forest(X_train, X_test, y_train, y_test)

        save_model(iso_model, 'isolation_forest_model.pkl')
        save_model(rf_model, 'random_forest_model.pkl')

        joblib.dump(scaler, 'demo/threat_hunting_ml/scaler.pkl')
        joblib.dump(le_event, 'demo/threat_hunting_ml/le_event.pkl')
        joblib.dump(le_user, 'demo/threat_hunting_ml/le_user.pkl')
        joblib.dump(le_ip, 'demo/threat_hunting_ml/le_ip.pkl')
        joblib.dump(features, 'demo/threat_hunting_ml/features.pkl')

        return jsonify({'status': 'success', 'message': 'Models trained successfully'})
    except Exception as e: 





        
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download_results')
def download_results():
    try:
        return send_file('demo/threat_hunting_ml/threat_detection_results.csv', as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/visualize')
def visualize():
    try:
        if not os.path.exists('demo/threat_hunting_ml/threat_detection_results.csv'):
            return "No results to visualize. Run threat detection first."

        results_df = pd.read_csv('demo/threat_hunting_ml/threat_detection_results.csv')

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Threat distribution
        threat_counts = results_df['is_threat'].value_counts()
        axes[0, 0].pie(threat_counts, labels=['Normal', 'Threat'], autopct='%1.1f%%')
        axes[0, 0].set_title('Threat Distribution')

        # Threats by event type
        threat_by_type = results_df[results_df['is_threat'] == 1]['event_type'].value_counts()
        threat_by_type.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Threats by Event Type')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Threats by hour
        threat_by_hour = results_df[results_df['is_threat'] == 1]['hour_of_day'].value_counts().sort_index()
        threat_by_hour.plot(kind='line', ax=axes[1, 0], marker='o')
        axes[1, 0].set_title('Threats by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Number of Threats')

        # Threat scores distribution
        results_df['combined_threat_score'].hist(ax=axes[1, 1], bins=20)
        axes[1, 1].set_title('Distribution of Combined Threat Scores')
        axes[1, 1].set_xlabel('Threat Score')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('demo/threat_hunting_ml/templates', exist_ok=True)
    os.makedirs('demo/threat_hunting_ml/static', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)
