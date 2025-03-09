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
        self.target_scaler = StandardScaler()  # Separate scaler for target
        self.models = {}
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df.columns = self.df.columns.str.strip()
        self.df.replace("-", np.nan, inplace=True)
        self.df["FastCharge_KmH"] = pd.to_numeric(self.df["FastCharge_KmH"], errors="coerce")
        
        # Handle missing values
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        self.df["Reviews"].fillna("No review available", inplace=True)

        # Sentiment Analysis
        self.df["SentimentScore"] = self.df["Reviews"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        
        # Encode RapidCharge (Yes/No → 1/0)
        self.df["RapidCharge"] = self.df["RapidCharge"].map({"Yes": 1, "No": 0})
        
        self.logger.info("Data loaded and preprocessed successfully.")

    def prepare_data(self):
        features = ["Range_Km", "Efficiency_WhKm", "FastCharge_KmH", "SentimentScore", "RapidCharge"]
        target = "PriceEuro"
        
        X = self.df[features]
        y = self.df[target].values.reshape(-1, 1)  # Reshape for scaler

        # Scale features and target
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y).flatten()

        return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    def train_models(self, X_train, y_train):
        self.models["KNN"] = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
        self.models["Random Forest"] = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        self.models["Linear Regression"] = LinearRegression().fit(X_train, y_train)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        # LSTM Model
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
        predictions = {}

        for name, model in self.models.items():
            if name == "LSTM":
                y_pred_scaled = model.predict(X_test_lstm).flatten()
            else:
                y_pred_scaled = model.predict(X_test)

            # Convert back to original scale
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            predictions[name] = y_pred

            results.append({
                "Model": name,
                "MAE": mean_absolute_error(self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), y_pred),
                "RMSE": np.sqrt(mean_squared_error(self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), y_pred)),
                "R²": r2_score(self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), y_pred)
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

        # Ensure y_test is scaled back before plotting
        y_test_original = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Plot sales predictions using SentimentPlotter (ensure this module exists)
        if "SentimentPlotter" in globals():
            SentimentPlotter.plot_sales_predictions(
                y_test_original, y_pred_knn, y_pred_rf, y_pred_lr, y_pred_lstm, benchmark_df
            )
        else:
            self.logger.warning("SentimentPlotter module not found, skipping plot.")

        # Display evaluation results
        print("\n🔹 Model Evaluation Results:")
        print(benchmark_df)

# Example execution:
if __name__ == "__main__":
    predictor = ElectricCarSentimentPrediction("ElectricCarData_Clean.csv")
    predictor.run()
