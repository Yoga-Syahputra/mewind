from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, -1])  
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, X_val, y_val):
    """
    Train the LSTM model using training and validation sequences.
    """
    # Build LSTM model
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(30),
        Dropout(0.2),
        Dense(1)  # Output satu nilai (FF_AVG)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

    # Evaluate on training set
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_val, y_pred_val)

    # Collect metrics for both training and validation
    metrics = {
        "Training": {"MAE": mae_train, "MSE": mse_train, "RMSE": rmse_train, "R2": r2_train},
        "Validation": {"MAE": mae_val, "MSE": mse_val, "RMSE": rmse_val, "R2": r2_val}
    }

    return model, metrics
