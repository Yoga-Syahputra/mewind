import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_svm_model(train_data, val_data):
    """Train the SVM model and evaluate metrics for training and validation."""
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]

    # Train the SVM model
    model = SVR(kernel='rbf', C=100, epsilon=0.1)
    model.fit(X_train, y_train)

    # Evaluate on training data
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Evaluate on validation data
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
