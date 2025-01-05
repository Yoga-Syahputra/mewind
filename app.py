from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from utils.preprocessing import preprocess_data, create_features, split_data, scale_data
from utils.svm_model import train_svm_model
from utils.lstm_model import create_sequences, train_lstm_model
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app 
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'data'
DATA_PATH = os.path.join('data', 'dataset.csv')
MODELS_PATH = os.path.join('models')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/components/training')
def training_process():
    return render_template('components/TrainingProcess.html')

@app.route('/components/testing')
def testing_process():
    return render_template('components/TestingProcess.html')

@app.route('/history', methods=['GET'])
def get_history():
    """Endpoint to serve the entire dataset."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce').dt.strftime('%Y-%m-%d')
            return jsonify(df.to_dict(orient='records'))
        else:
            return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        file.save(DATA_PATH)
        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

@app.route('/dataset', methods=['GET'])
def get_dataset():
    """Endpoint to serve the entire dataset."""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)

            # Format the date column
            if 'TANGGAL' in df.columns:
                df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce').dt.strftime('%Y-%m-%d')

            # Convert entire dataset to JSON
            return jsonify(df.to_dict(orient='records'))
        else:
            return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train/<model_type>', methods=['POST'])
def train_model(model_type):
    """Train the specified model (SVM or LSTM)."""
    try:
        # Load dataset
        if not os.path.exists(DATA_PATH):
            return jsonify({'error': 'Dataset not found. Please upload a dataset first.'}), 404

        df = pd.read_csv(DATA_PATH)

        # Preprocess data
        if 'TANGGAL' in df.columns:
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
        df.dropna(subset=['TANGGAL'], inplace=True)
        df = preprocess_data(df)
        df = create_features(df)

        # Define feature columns
        feature_columns = ['TX', 'RH_AVG', 'RR', 'FF_X']

        # Split and scale data
        train_data, val_data, _ = split_data(df)
        train_scaled, val_scaled, _, scaler = scale_data(train_data, val_data, None, feature_columns)

        # Train model
        if model_type == 'lstm':
            X_train, y_train = create_sequences(train_scaled, seq_length=3)
            X_val, y_val = create_sequences(val_scaled, seq_length=3)
            model, metrics = train_lstm_model(X_train, y_train, X_val, y_val)
            model.save(os.path.join(MODELS_PATH, 'lstm_model.h5'))
        elif model_type == 'svm':
            model, metrics = train_svm_model(train_scaled, val_scaled)
            joblib.dump(model, os.path.join(MODELS_PATH, 'svm_model.pkl'))
        else:
            return jsonify({'error': 'Invalid model type. Use "svm" or "lstm".'}), 400

        # Save scaler
        joblib.dump(scaler, os.path.join(MODELS_PATH, 'scaler.pkl'))

        return jsonify({'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
     
    
@app.route('/predict/<model_type>', methods=['POST'])
def predict_model(model_type):
    """Predict wind speed using the specified model."""
    try:
        # Load scaler
        scaler_path = os.path.join(MODELS_PATH, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({'error': 'Scaler not found. Train the model first.'}), 404
        scaler = joblib.load(scaler_path)

        # Get user input
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400

        # Prepare input data
        feature_columns = ['TX', 'RH_AVG', 'RR', 'FF_X']
        if model_type == 'svm':
            input_data = [[
                data.get('temperature', 0),
                data.get('humidity', 0),
                data.get('rainfall', 0),
                data.get('wind_gust', 0)
            ]]

            # Validate feature length
            if len(input_data[0]) != len(feature_columns):
                return jsonify({'error': f"Expected {len(feature_columns)} features, but got {len(input_data[0])}."}), 400

            # Scale input data
            input_scaled = scaler.transform(input_data)

            # Load SVM model and predict
            model_path = os.path.join(MODELS_PATH, 'svm_model.pkl')
            if not os.path.exists(model_path):
                return jsonify({'error': 'SVM model not found. Train the model first.'}), 404
            model = joblib.load(model_path)
            prediction = float(model.predict(input_scaled)[0])  # Ensure float for JSON

        elif model_type == 'lstm':
            seq_length = 3
            historical_data = data.get('historical_data', [])
            if len(historical_data) < seq_length:
                return jsonify({'error': f'LSTM requires at least {seq_length} days of historical data.'}), 400

            input_data = [[
                entry.get('temperature', 0),
                entry.get('humidity', 0),
                entry.get('rainfall', 0),
                entry.get('wind_gust', 0)
            ] for entry in historical_data[-seq_length:]]

            # Validate feature length
            if any(len(row) != len(feature_columns) for row in input_data):
                return jsonify({'error': 'Invalid number of features in historical data.'}), 400

            # Scale and reshape data for LSTM
            input_scaled = scaler.transform(input_data).reshape(1, seq_length, len(feature_columns))

            # Load LSTM model and predict
            model_path = os.path.join(MODELS_PATH, 'lstm_model.h5')
            if not os.path.exists(model_path):
                return jsonify({'error': 'LSTM model not found. Train the model first.'}), 404
            model = load_model(model_path)
            prediction = float(model.predict(input_scaled)[0][0])  # Ensure float for JSON

        else:
            return jsonify({'error': 'Invalid model type. Use "svm" or "lstm".'}), 400

        # Return prediction
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)