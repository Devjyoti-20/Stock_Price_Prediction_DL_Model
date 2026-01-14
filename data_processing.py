# data_processing.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta

def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_and_split_data(data, test_size=0.2, val_size=0.25, scaler_path=".//models//dataset_scaler.gz"):
    features = ["Open", "High", "Low", "Close", "Volume"]
    
    # Ensure date is in datetime format
    data["Date"] = pd.to_datetime(data["Date"])

    # Split into train+val and test
    train_val_data, test_data, train_val_dates, test_dates = train_test_split(
        data[features], data["Date"], test_size=test_size, shuffle=False
    )

    # Split train_val into train and validation
    train_data, val_data, train_dates, val_dates = train_test_split(
        train_val_data, train_val_dates, test_size=val_size, shuffle=False
    )

    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # Save scaler model
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Combine with dates for saving
    train_df = pd.DataFrame(train_scaled, columns=features)
    train_df["Date"] = train_dates.values

    val_df = pd.DataFrame(val_scaled, columns=features)
    val_df["Date"] = val_dates.values

    test_df = pd.DataFrame(test_scaled, columns=features)
    test_df["Date"] = test_dates.values

    return train_df, val_df, test_df

def save_split_datasets(train_df, val_df, test_df, data_dir=".//data//"):
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "validate.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
def load_datasets():
    data_file_location = ".//data//"
    data_file_ext = "csv"

    train_df = pd.read_csv(data_file_location + "train." + data_file_ext)
    validate_df = pd.read_csv(data_file_location + "validate." + data_file_ext)
    test_df = pd.read_csv(data_file_location + "test." + data_file_ext)

    train_df["Date"] = pd.to_datetime(train_df["Date"])
    validate_df["Date"] = pd.to_datetime(validate_df["Date"])
    test_df["Date"] = pd.to_datetime(test_df["Date"])

    return train_df, validate_df, test_df


def extract_features(data_train_df, data_validate_df, data_test_df):
    features = ["Open", "High", "Low", "Close", "Volume"]

    data_train_scaled = data_train_df[features].values
    data_validate_scaled = data_validate_df[features].values
    data_test_scaled = data_test_df[features].values

    return data_train_scaled, data_validate_scaled, data_test_scaled


def construct_lstm_data(data, sequence_size, target_attr_idx):
    data_X, data_y = [], []
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i, 0:data.shape[1]])
        data_y.append(data[i, target_attr_idx])
    return np.array(data_X), np.array(data_y)


def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(GRU(units=100, return_sequences=True)))
    for _ in range(3):
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(rate=0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    model_location = ".//models//"
    model_name = "stock_price.model.keras"
    checkpoint = ModelCheckpoint(
        model_location + model_name, monitor="val_loss",
        save_best_only=True, mode="min", verbose=0
    )
    history = model.fit(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[checkpoint]
    )
    return history


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2, y_pred


def load_scaler():
    scaler_model_location = ".//models//"
    scaler_model_name = "dataset_scaler.gz"
    sc = joblib.load(scaler_model_location + scaler_model_name)
    return sc


def inverse_transform_predictions(scaler, y_data, y_pred):
    filler = np.ones((len(y_data), 4))
    y_data_inv = scaler.inverse_transform(np.concatenate((y_data.reshape(-1,1), filler), axis=1))[:,0]
    y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, filler), axis=1))[:,0]
    return y_data_inv, y_pred_inv

def add_technical_indicators(df):
    df = df.copy()
    df.fillna(method='bfill', inplace=True)
    # Simple Moving Average (SMA)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    
    # Exponential Moving Average (EMA)
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + RS))
    
    # MACD
    EMA_12 = df["Close"].ewm(span=12, adjust=False).mean()
    EMA_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = EMA_12 - EMA_26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    return df

def plot_predictions(dates_train, dates_val, dates_test,
                     y_train_actual, y_train_pred,
                     y_val_actual, y_val_pred,
                     y_test_actual, y_test_pred, ticker):
    plt.figure(figsize=(18,6))
    plt.plot(dates_train, y_train_actual, label="Training Data", color="lightblue")
    plt.plot(dates_train, y_train_pred, label="Training Predictions", linewidth=1, color="violet")
    plt.plot(dates_val, y_val_actual, label="Validation Data", color="yellow")
    plt.plot(dates_val, y_val_pred, label="Validation Predictions", linewidth=1, color="red")
    plt.plot(dates_test, y_test_actual, label="Testing Data", color="lightgreen")
    plt.plot(dates_test, y_test_pred, label="Testing Predictions", linewidth=1, color="green")
    plt.title(f"{ticker} Stock Price Predictions With BiGRU+LSTM")
    plt.xlabel("Time (YYYY-MM)")
    plt.ylabel("Stock Price (USD)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(color="lightgray")
    return plt
