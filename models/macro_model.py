# macroeconomic status model

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

class MacroModel:
    def __init__(self, model_path="models/saved_macro_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def load_data(self, macro_data_path, stock_data_path):
        df_macro = pd.read_csv(macro_data_path, parse_dates=['Date'], index_col='Date')
        df_stock = pd.read_csv(stock_data_path, parse_dates=['Date'], index_col='Date')
        df_macro = df_macro.resample("D").ffill()

        df = df_stock.join(df_macro, how="inner")
        df.dropna(inplace=True)

        df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
        df.dropna(inplace=True)

        return df
    
    def preprocess(self, df):
        X = df.drop(columns=["Target", "Close"])
        y = df["Target"]
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("AUC Score:", roc_auc_score(y_test, y_proba))

    def save_model(self):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_path)

    def load_model(self):
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]

    def predict(self, latest_macro_data):
        latest_scaled = self.scaler.transform(latest_macro_data)
        proba = self.model.predict_proba(latest_scaled)[:, 1]
        return proba[0]

def get_macro_signal(ticker: str):
    try:
        model = MacroModel()
        model.load_model()

        macro_path = "data/macro_data/macro_features.csv"
        price_path = f"data/price_data/{ticker}_price_data.csv"

        df = model.load_data(macro_path, price_path)
        latest_row = df.iloc[[-1]].drop(columns=["Target", "Close"])
        proba = model.predict(latest_row)

        signal = "buy" if proba > 0.6 else "sell" if proba < 0.4 else "hold"
        return {
            "model_name": "macro",
            "signal": signal,
            "confidence": round(abs(proba - 0.5) * 2, 3)  # 0 to 1 scale
        }

    except Exception as e:
        print(f"[Macro Model Error] {e}")
        return {"model_name": "macro", "signal": "hold", "confidence": 0.0}


if __name__ == "__main__":
    model = MacroModel()

    df = model.load_data("data/macro_data/macro_features.csv", "data/price_data/AAPL_price_data.csv")
    X_train, X_test, y_train, y_test = model.preprocess(df)

    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save_model()
