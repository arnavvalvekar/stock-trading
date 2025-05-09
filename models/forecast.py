# time series forecasting model

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# load and preprocess data
def load_price_data(file_path, features, target_col='Close', window=60, forecast_horizon=1):
    df = pd.read_csv(file_path)
    df = engineer_features(df)
    df = df[features + [target_col]].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - window - forecast_horizon):
        X.append(scaled[i:i+window, :-1])
        y.append(scaled[i+window:i+window+forecast_horizon, -1])  # Multi-step output
    return np.array(X), np.array(y), scaler

# feature engineering
def engineer_features(df):
    df = df.copy()
    df["50_MA"] = df["Close"].rolling(window=50).mean()
    df["200_MA"] = df["Close"].rolling(window=200).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # bollinger bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    # MACD
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

# define LSTM model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, seq_len, forecast_horizon, num_filters=32, kernel_size=3, hidden_size=64, num_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)  # -> [batch_size, input_size, seq_len]
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # -> [batch_size, seq_len', num_filters]
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
    
# train model
def train_model(model, train_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")


# plot results
def evaluate(model, test_loader, y_test):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = model(X_batch)
            preds.extend(pred.cpu().numpy())

    preds = np.array(preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Evaluation Metrics:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100, 0], label='Actual (t+1)')
    plt.plot(preds[:100, 0], label='Predicted (t+1)')
    plt.legend()
    plt.title('Forecast (First Horizon)')
    plt.show()

def load_model(path, input_size, seq_len, forecast_horizon):
    model = CNNLSTMModel(input_size=input_size, seq_len=seq_len, forecast_horizon=forecast_horizon)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_stock(model, X_latest):  # X_latest = latest window of input
    with torch.no_grad():
        pred = model(torch.Tensor(X_latest).unsqueeze(0))
        return pred.numpy().flatten()

def get_cnn_signal(ticker: str, forecast_horizon=1, window=60):
    file_path = os.path.join("data", "price_data", f"{ticker}_price_data.csv")
    
    features = ['Close', 'RSI', 'MACD', 'MACD_Signal', "Volume"]
    target = 'Close'

    try:
        X, y, scaler = load_price_data(file_path, features, target_col=target, window=window, forecast_horizon=forecast_horizon)

        X_latest = X[-1]
        current_price = scaler.inverse_transform(np.concatenate([X_latest[-1], [0]]).reshape(1, -1))[0][0]  # unscale 'Close'

        model_path = os.path.join("models", f"{ticker}_cnn.pt")  # update if different
        model = load_model(model_path, input_size=X.shape[2], seq_len=X.shape[1], forecast_horizon=forecast_horizon)

        forecast = predict_stock(model, X_latest)
        predicted_price = scaler.inverse_transform(np.concatenate([X_latest[-1], [forecast[0]]]).reshape(1, -1))[0][-1]

        signal = "buy" if predicted_price > current_price * 1.01 else "sell" if predicted_price < current_price * 0.99 else "hold"
        confidence = abs(predicted_price - current_price) / current_price  # relative confidence

        return {
            "model_name": "cnn",
            "signal": signal,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        print(f"[CNN Model Error] Failed to generate signal for {ticker}: {e}")
        return {
            "model_name": "cnn",
            "signal": "hold",
            "confidence": 0.0
        }

if __name__ == "__main__":
    price_path = os.path.join("data", "price_data", "AAPL_price_data.csv")

    features = ['Close', 'RSI', 'MACD', 'MACD_Signal', "Volume"]
    target = 'Close'
    forecast_horizon = 1
    window = 60

    X, y, scaler = load_price_data(price_path, features, target_col=target, window=window, forecast_horizon=forecast_horizon)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)), batch_size=64, shuffle=False)

    model = CNNLSTMModel(input_size=X.shape[2], seq_len=X.shape[1], forecast_horizon=forecast_horizon)
    train_model(model, train_loader)
    evaluate(model, test_loader, y_test)
