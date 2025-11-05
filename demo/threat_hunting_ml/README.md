# Threat Hunting using Machine Learning on Security Event Logs

This project implements a machine learning-based threat hunting system for analyzing security event logs. It uses anomaly detection and classification techniques to identify potential security threats in log data.

## Features

- **Data Generation**: Simulate realistic security event logs with normal and anomalous patterns
- **Machine Learning Models**:
  - Isolation Forest for unsupervised anomaly detection
  - Random Forest Classifier for supervised threat classification
  - Ensemble approach combining both models
- **Web Dashboard**: Interactive Flask-based web interface for data analysis and visualization
- **Real-time Threat Detection**: Process new log entries and score them for threats
- **Visualization**: Charts and graphs showing threat patterns and distributions

## Project Structure

```
threat_hunting_ml/
├── data_generator.py      # Generate simulated security logs
├── preprocess.py          # Data preprocessing and feature engineering
├── model.py              # ML model training and evaluation
├── detect.py             # Threat detection engine
├── app.py                # Flask web application
├── templates/
│   └── index.html        # Web dashboard template
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── static/              # Static files (CSS, JS, images)
```

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line

1. **Generate sample data**:
   ```bash
   python data_generator.py
   ```

2. **Train ML models**:
   ```bash
   python model.py
   ```

3. **Run threat detection**:
   ```bash
   python detect.py
   ```

### Web Interface

1. Start the web application:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:5000`

3. Use the dashboard to:
   - Generate sample security logs
   - Train machine learning models
   - Run threat detection analysis
   - View visualizations
   - Download results

## Machine Learning Approach

### Models Used

1. **Isolation Forest**:
   - Unsupervised anomaly detection
   - Effective for detecting novel threats
   - Uses isolation trees to identify outliers

2. **Random Forest Classifier**:
   - Supervised classification
   - Learns patterns from labeled data
   - Provides probability scores for threats

3. **Ensemble Method**:
   - Combines predictions from both models
   - Uses weighted average of anomaly scores and classification probabilities
   - Improves overall detection accuracy

### Features Engineered

- Event type encoding
- User and IP address encoding
- Temporal features (hour, day of week, month)
- Session and network metrics
- Login attempt patterns

## Threat Detection Process

1. **Data Ingestion**: Load security logs from CSV or real-time sources
2. **Preprocessing**: Clean, encode, and scale features
3. **Model Scoring**: Apply both ML models to score each log entry
4. **Ensemble Decision**: Combine model outputs with configurable threshold
5. **Alert Generation**: Flag high-confidence threats for investigation

## Evaluation Metrics

- Precision: Ability to avoid false positives
- Recall: Ability to detect actual threats
- AUC-ROC: Overall model discrimination ability
- F1-Score: Balance between precision and recall

## Security Considerations

- Models are trained on simulated data; real deployment requires domain-specific training
- Regularly retrain models with new threat patterns
- Implement proper access controls for the web interface
- Monitor model performance and update thresholds as needed

## Future Enhancements

- Integration with SIEM systems (Splunk, ELK Stack)
- Real-time streaming data processing
- Deep learning models (LSTM, Autoencoders)
- Multi-class threat categorization
- Automated response actions
- API endpoints for external integrations

## License

This project is open-source and available under the MIT License.
