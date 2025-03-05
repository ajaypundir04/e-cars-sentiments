import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from textblob import TextBlob
import logging

from output.sentiment_plotter import SentimentPlotter  # Ensure this module exists or replace it

class ElectricCarSentimentPrediction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
        self.models = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df.columns = self.df.columns.str.strip()
        self.df.replace("-", np.nan, inplace=True)
        self.df["FastCharge_KmH"] = pd.to_numeric(self.df["FastCharge_KmH"], errors="coerce")
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        self.df["Reviews"].fillna("No review available", inplace=True)
        self.df["SentimentScore"] = self.df["Reviews"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        self.df["RapidCharge"] = self.df["RapidCharge"].map({"Yes": 1, "No": 0})
        self.logger.info("Data loaded and preprocessed successfully.")

    def prepare_data(self):
        features = ["Range_Km", "Efficiency_WhKm", "FastCharge_KmH", "SentimentScore", "RapidCharge"]
        target = "PriceEuro"
        X = self.df[features]
        y = self.df[target]
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_models(self, X_train, y_train):
        self.models["KNN"] = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
        self.models["Random Forest"] = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        self.models["Linear Regression"] = LinearRegression().fit(X_train, y_train)
        
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        lstm.compile(optimizer="adam", loss="mse")
        lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=16, verbose=1)
        self.models["LSTM"] = lstm
        self.logger.info("All models trained successfully.")

    def evaluate_models(self, X_test, y_test):
        X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        results = []
        predictions = {}  # Store predictions for plotting

        for name, model in self.models.items():
            if name == "LSTM":
                y_pred = model.predict(X_test_lstm).flatten()
            else:
                y_pred = model.predict(X_test)
            
            predictions[name] = y_pred  # Store predictions for this model
            
            results.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred)
            })
    
        return pd.DataFrame(results), predictions

    def run(self):
        self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.train_models(X_train, y_train)
        benchmark_df, predictions = self.evaluate_models(X_test, y_test)
        
        # Extract individual predictions
        y_pred_knn = predictions["KNN"]
        y_pred_rf = predictions["Random Forest"]
        y_pred_lr = predictions["Linear Regression"]
        y_pred_lstm = predictions["LSTM"]
        
        # Plot sales predictions using SentimentPlotter (ensure this module exists)
        if 'SentimentPlotter' in globals():
            SentimentPlotter.plot_sales_predictions(y_test, y_pred_knn, y_pred_rf, y_pred_lr, y_pred_lstm, benchmark_df)
        else:
            print("SentimentPlotter module not found, skipping plot.")

        # Display evaluation results
        print(benchmark_df)

# Example execution:
if __name__ == "__main__":
    predictor = ElectricCarSentimentPrediction("ElectricCarData_Clean.csv")
    predictor.run()
