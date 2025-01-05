import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """Preprocess the dataset."""
    # Replace invalid data with NaN
    df.replace([8888, 9999, '-'], np.nan, inplace=True)

    # Convert numeric columns
    numeric_columns = ['TX', 'RH_AVG', 'RR', 'FF_X', 'FF_AVG']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN with 0
    df.dropna(inplace=True)

    # Handle missing values using interpolation
    df.interpolate(method='linear', inplace=True)  

    # Replace 0 wind speed with a small value
    df['FF_AVG'] = df['FF_AVG'].replace(0, df['FF_AVG'].mean())

    # Log-transform RR and FF_AVG to reduce skewness
    df['RR'] = np.log1p(df['RR'])
    df['FF_AVG'] = np.log1p(df['FF_AVG'])

    return df

def split_data(df, train_size=0.7, val_size=0.2):
    """Split the data into train, validation, and test sets."""
    total_size = len(df)
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))
    return df[:train_end], df[train_end:val_end], df[val_end:]

def scale_data(train_data, val_data, test_data, feature_columns):
    """Scale the data using MinMaxScaler."""
    scaler = MinMaxScaler()
    train_scaled = val_scaled = test_scaled = None

    if train_data is not None:
        train_scaled = scaler.fit_transform(train_data[feature_columns])
    if val_data is not None:
        val_scaled = scaler.transform(val_data[feature_columns])
    if test_data is not None:
        test_scaled = scaler.transform(test_data[feature_columns])

    return train_scaled, val_scaled, test_scaled, scaler
