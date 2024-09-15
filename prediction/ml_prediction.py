import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, LayerNormalization, MultiHeadAttention, Input, Dropout
from keras import Model
from utils.log_utils import LoggerManager

class ECarSentimentPrediction:
    def __init__(self, model_type='all', log_level=logging.INFO):
        """
        Initialize the ECarSentimentPrediction class with specified model type and logger.
        Args:
            model_type (str): The type of model to use ('linear_regression', 'knn', 'random_forest', 'lstm', 'transformer', or 'all').
            log_level (int): Logging level.
        """
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        # Initialize models
        self.linear_reg = LinearRegression()
        self.knn = KNeighborsRegressor()
        self.random_forest = RandomForestRegressor()
        self.lstm = None
        self.transformer = None
        self.model_type = model_type
        self.logger.info(f"Initialized with model type: {self.model_type}")
    
    def prepare_data(self, sentiment_data):
        """
        Prepare sentiment data for model training.
        Args:
            sentiment_data (DataFrame): Sentiment scores with corresponding target variables.
        Returns:
            X (DataFrame): Features.
            y (Series): Target variable.
        """
        X = sentiment_data[['positive_count', 'negative_count', 'neutral_count']]
        y = sentiment_data['future_sales']
        self.logger.info("Data prepared for model training.")
        return X, y

    def train_linear_regression(self, X, y):
        self.linear_reg.fit(X, y)
        self.logger.info("Trained Linear Regression model.")

    def train_knn(self, X, y, n_neighbors=5):
        if len(X) < n_neighbors:
            n_neighbors = len(X)
            self.logger.warning(f"Not enough samples for KNN with n_neighbors={n_neighbors}. Using n_neighbors={n_neighbors} instead.")
        
        self.knn.set_params(n_neighbors=n_neighbors)
        self.knn.fit(X, y)
        self.logger.info(f"Trained KNN model with n_neighbors={n_neighbors}.")

    def train_random_forest(self, X, y):
        self.random_forest.fit(X, y)
        self.logger.info("Trained Random Forest model.")

    def train_lstm(self, X, y, input_shape):
        """
        Train an LSTM model.
        Args:
            X (DataFrame): Feature data.
            y (Series): Target data.
            input_shape (tuple): Input shape required for LSTM.
        """
        # Convert X to a NumPy array and reshape it for LSTM
        X = X.values.reshape((X.shape[0], X.shape[1], 1))
        
        # Reshape y to be a 2D array (required by Keras)
        y = y.values.reshape(-1, 1)

        self.lstm = Sequential()
        self.lstm.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        self.lstm.add(LSTM(units=50))
        self.lstm.add(Dense(1))
        self.lstm.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm.fit(X, y, epochs=50, batch_size=32)

    def train_transformer(self, X, y, input_shape):
        """
        Train a Transformer-based model.
        Args:
            X (DataFrame): Feature data.
            y (Series): Target data.
            input_shape (tuple): Input shape required for Transformer.
        """
        X = X.values.reshape((X.shape[0], X.shape[1], 1))
        y = y.values.reshape(-1, 1)

        inputs = Input(shape=input_shape)
        x = MultiHeadAttention(num_heads=2, key_dim=64)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(1)(x)

        self.transformer = Model(inputs, x)
        self.transformer.compile(optimizer='adam', loss='mean_squared_error')
        self.transformer.fit(X, y, epochs=50, batch_size=32)

    def train_all_models(self, X, y):
        """
        Train all models (Linear Regression, KNN, Random Forest, LSTM, Transformer).
        """
        self.train_linear_regression(X, y)
        self.train_knn(X, y)
        self.train_random_forest(X, y)
        input_shape = (X.shape[1], 1)
        self.train_lstm(X, y, input_shape)
        self.train_transformer(X, y, input_shape)

    def predict_future(self, model, X):
        self.logger.info(f"Predicting future using {model} model.")
        if model == 'linear_regression':
            return self.linear_reg.predict(X)
        elif model == 'knn':
            return self.knn.predict(X)
        elif model == 'random_forest':
            return self.random_forest.predict(X)
        elif model == 'lstm':
            X = X.values  # Convert DataFrame to NumPy array
            X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
            return self.lstm.predict(X)
        elif model == 'transformer':
            X = X.values  # Convert DataFrame to NumPy array
            X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for Transformer
            return self.transformer.predict(X)

    def run(self, X, y):
        """
        Run the training and prediction based on the selected model type.
        """
        if self.model_type == 'linear_regression':
            self.train_linear_regression(X, y)
        elif self.model_type == 'knn':
            self.train_knn(X, y)
        elif self.model_type == 'random_forest':
            self.train_random_forest(X, y)
        elif self.model_type == 'lstm':
            input_shape = (X.shape[1], 1)
            self.train_lstm(X, y, input_shape)
        elif self.model_type == 'transformer':
            input_shape = (X.shape[1], 1)
            self.train_transformer(X, y, input_shape)
        elif self.model_type == 'all':
            self.train_all_models(X, y)

        predictions = {}
        for model in ['linear_regression', 'knn', 'random_forest', 'lstm', 'transformer']:
            predictions[model] = self.predict_future(model, X)

        # Convert predictions to a readable format
        readable_predictions = {
            'Linear Regression Prediction': round(float(predictions['linear_regression'][0]), 2),
            'K-Nearest Neighbors Prediction': round(float(predictions['knn'][0]), 2),
            'Random Forest Prediction': round(float(predictions['random_forest'][0]), 2),
            'LSTM Prediction': round(float(predictions['lstm'][0][0]), 2),  # Access the first element in the LSTM output
            'Transformer Prediction': round(float(predictions['transformer'][0][0]), 2)  # Access the first element in the Transformer output
        }

        readable_output = "\n".join([f"{model}: {value}" for model, value in readable_predictions.items()])
        self.logger.info(f"Predictions:\n{readable_output}")

        return readable_predictions
