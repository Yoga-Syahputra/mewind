from flask import Flask, request, jsonify, render_template 
import os
import pandas as pd
from utils.preprocessing import preprocess_data, split_data, scale_data
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

        # Define feature columns
        feature_columns = ['TX', 'RH_AVG', 'RR', 'FF_X']

        # Split data
        train_data, val_data, _ = split_data(df)


        if model_type == 'lstm':
            # Scale data for LSTM
            train_scaled, val_scaled, _, lstm_scaler = scale_data(train_data, val_data, None, feature_columns)
            X_train, y_train = create_sequences(train_scaled, seq_length=3)
            X_val, y_val = create_sequences(val_scaled, seq_length=3)

            # Train LSTM model
            model, metrics = train_lstm_model(X_train, y_train, X_val, y_val)
            model.save(os.path.join(MODELS_PATH, 'lstm_model.h5'))

            # Save LSTM scaler
            joblib.dump(lstm_scaler, os.path.join(MODELS_PATH, 'lstm_scaler.pkl'))

        elif model_type == 'svm':
            # Scale data for SVM
            train_scaled, val_scaled, _, svm_scaler = scale_data(train_data, val_data, None, feature_columns)

            # Train SVM model
            model, metrics = train_svm_model(train_scaled, val_scaled)
            joblib.dump(model, os.path.join(MODELS_PATH, 'svm_model.pkl'))

            # Save SVM scaler
            joblib.dump(svm_scaler, os.path.join(MODELS_PATH, 'svm_scaler.pkl'))

        else:
            return jsonify({'error': 'Invalid model type. Use "svm" or "lstm".'}), 400

        return jsonify({'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<model_type>', methods=['POST'])
def predict_model(model_type):
    """Predict wind speed for multiple days using the specified model."""
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

        feature_columns = ['TX', 'RH_AVG', 'RR', 'FF_X']

        if model_type == 'svm':
            # Load dataset for historical data (if necessary for features, else skip this)
            if not os.path.exists(DATA_PATH):
                return jsonify({'error': 'Dataset not found. Please upload a dataset first.'}), 404

            df = pd.read_csv(DATA_PATH)
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
            df = df.sort_values('TANGGAL')

            # Extract historical data for visualization (not for prediction)
            forecast_date = pd.to_datetime(data.get('forecast_date'))
            if not forecast_date:
                return jsonify({'error': 'Forecast date is required.'}), 400

            historical_data = df[df['TANGGAL'] < forecast_date].tail(2)
            if len(historical_data) < 2:
                # If no historical data, return default empty values for visualization
                historical_data = pd.DataFrame({
                    'TANGGAL': [forecast_date - pd.Timedelta(days=2), forecast_date - pd.Timedelta(days=1)],
                    'FF_X': [0, 0]
                })

            # Prepare input for SVM prediction
            input_data = [[
                data.get('temperature', 0),
                data.get('humidity', 0),
                data.get('rainfall', 0),
                data.get('wind_gust', 0)
            ]]
            input_scaled = scaler.transform(input_data)

            # Load SVM model
            model_path = os.path.join(MODELS_PATH, 'svm_model.pkl')
            if not os.path.exists(model_path):
                return jsonify({'error': 'SVM model not found. Train the model first.'}), 404
            model = joblib.load(model_path)

            # Predict for the given input
            prediction = float(model.predict(input_scaled)[0])

            # Combine historical data with the new prediction
            dates = list(historical_data['TANGGAL'].dt.strftime('%Y-%m-%d'))
            dates.append(forecast_date.strftime('%Y-%m-%d'))

            historical_predictions = list(historical_data['FF_X'])  # Example feature for visualization
            predictions = historical_predictions + [prediction]

            response = [{'date': date, 'prediction': pred} for date, pred in zip(dates, predictions)]

        elif model_type == 'lstm':
            # LSTM Prediction
            forecast_date = pd.to_datetime(data.get('forecast_date'))
            if not forecast_date:
                return jsonify({'error': 'Forecast date is required.'}), 400

            if not os.path.exists(DATA_PATH):
                return jsonify({'error': 'Dataset not found. Please upload a dataset first.'}), 404

            df = pd.read_csv(DATA_PATH)
            df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
            df = df.sort_values('TANGGAL')

            # Extract the last 3 days up to the forecast date
            sequence_data = df[df['TANGGAL'] <= forecast_date].tail(3)
            if len(sequence_data) < 3:
                return jsonify({'error': 'Not enough data for the required sequence.'}), 400

            input_data = sequence_data[feature_columns].values
            input_scaled = scaler.transform(input_data).reshape(1, 3, len(feature_columns))

            # Load LSTM model
            model_path = os.path.join(MODELS_PATH, 'lstm_model.h5')
            if not os.path.exists(model_path):
                return jsonify({'error': 'LSTM model not found. Train the model first.'}), 404
            model = load_model(model_path)

            # Predict the next day based on the input sequence
            prediction = float(model.predict(input_scaled)[0][0])  # LSTM prediction

            # Prepare response with the last two days and the predicted day
            dates = list(sequence_data['TANGGAL'].dt.strftime('%Y-%m-%d'))
            dates.append(forecast_date.strftime('%Y-%m-%d'))
            predictions = list(sequence_data['FF_X'])  # Example feature for historical data
            predictions.append(prediction)

            response = [{'date': date, 'prediction': pred} for date, pred in zip(dates, predictions)]

        else:
            return jsonify({'error': 'Invalid model type. Use "svm" or "lstm".'}), 400

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
