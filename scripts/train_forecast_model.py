import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import finnhub
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

class CNNForecastModel(nn.Module):
    def __init__(self, input_size=1, seq_len=60, forecast_horizon=1):
        super(CNNForecastModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        
        # Calculate the size after convolutions
        conv_output_size = seq_len - 6  # 3 layers of conv with kernel_size=3
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, forecast_horizon)
        
        # Dropout and activation
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def prepare_data(data, seq_len=60):
    """Prepare data for training."""
    # Normalize data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(normalized_data) - seq_len):
        X.append(normalized_data[i:(i + seq_len)])
        y.append(normalized_data[i + seq_len])
    
    return np.array(X), np.array(y), scaler

def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """Train the model."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

def get_historical_data(ticker, days=365):
    """Get historical data using daily quotes."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get company profile for timezone
    profile = finnhub_client.company_profile2(symbol=ticker)
    if not profile:
        return None
    
    # Get daily quotes
    quotes = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            quote = finnhub_client.quote(ticker)
            if quote and 'c' in quote:  # 'c' is current price
                quotes.append(quote['c'])
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            current_date += timedelta(days=1)
            
        except Exception as e:
            print(f"Error fetching quote for {ticker} on {current_date}: {str(e)}")
            time.sleep(1)
            continue
    
    return np.array(quotes) if quotes else None

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Get training data
    print("Fetching training data...")
    
    # Get data for multiple stocks to train on
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Major tech stocks
    all_data = []
    
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = get_historical_data(ticker)
            if data is not None:
                all_data.extend(data)
                print(f"Got {len(data)} data points for {ticker}")
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    if not all_data:
        raise ValueError("No training data available")
    
    print(f"Total data points collected: {len(all_data)}")
    
    # Prepare data
    print("Preparing data...")
    X, y, scaler = prepare_data(np.array(all_data))
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNForecastModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save model and scaler
    print("Saving model...")
    model_path = models_dir / "forecast_model.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, model_path)
    
    print(f"Model saved to {model_path}")
    print("Training complete!")

if __name__ == "__main__":
    main() 