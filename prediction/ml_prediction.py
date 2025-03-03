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

# Load dataset
file_path = "ElectricCarData_Clean.csv"
df = pd.read_csv(file_path)

#  Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

#  Replace invalid values ("-") with NaN
df.replace("-", np.nan, inplace=True)

#  Ensure FastCharge_KmH is numeric
df["FastCharge_KmH"] = pd.to_numeric(df["FastCharge_KmH"], errors="coerce")

#  Ensure only numeric columns get mean filling
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

#  Fill missing review texts
df["Reviews"].fillna("No review available", inplace=True)

#  Sentiment Analysis Function
def get_sentiment(review):
    analysis = TextBlob(str(review))  # Convert to string to avoid NaN issues
    return analysis.sentiment.polarity  # Score between -1 (negative) and 1 (positive)

#  Apply sentiment analysis
df["SentimentScore"] = df["Reviews"].apply(get_sentiment)

#  Encode categorical features
df["RapidCharge"] = df["RapidCharge"].map({"Yes": 1, "No": 0})  # Convert Yes/No to 1/0

#  Select only numerical features
features = ["Range_Km", "Efficiency_WhKm", "FastCharge_KmH", "SentimentScore", "RapidCharge"]
target = "PriceEuro"  # Predicting sales as a proxy for market success

X = df[features]
y = df[target]

#  Normalize numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Initialize models
knn = KNeighborsRegressor(n_neighbors=5)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()

#  Train models
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

#  LSTM Model Preparation
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=16, verbose=1)

#  Predictions
y_pred_knn = knn.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_lstm = lstm.predict(X_test_lstm).flatten()

#  Benchmarking: Compute Errors
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R²": r2}

#  Compare Models
benchmark_results = [
    evaluate_model(y_test, y_pred_knn, "KNN"),
    evaluate_model(y_test, y_pred_rf, "Random Forest"),
    evaluate_model(y_test, y_pred_lr, "Linear Regression"),
    evaluate_model(y_test, y_pred_lstm, "LSTM")
]

benchmark_df = pd.DataFrame(benchmark_results)

#  Print Predictions Before Plotting
print("\n🔹 Model Predictions vs Actual Sales:\n")
print(f"KNN Predictions:\n{y_pred_knn[:5]}")
print(f"Random Forest Predictions:\n{y_pred_rf[:5]}")
print(f"Linear Regression Predictions:\n{y_pred_lr[:5]}")
print(f"LSTM Predictions:\n{y_pred_lstm[:5]}")
print("\n🔹 Benchmark Results:\n")
print(benchmark_df)

#  Plot Benchmark Results
plt.figure(figsize=(10, 5))
sns.barplot(data=benchmark_df.melt(id_vars=["Model"]), x="Model", y="value", hue="variable")
plt.title("Benchmark Comparison of ML Models")
plt.xlabel("Model")
plt.ylabel("Error Metrics")
plt.show()

#  Scatter Plot of Actual vs. Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Sales", marker="o")
plt.plot(y_pred_knn, label="KNN Prediction", linestyle="--")
plt.plot(y_pred_rf, label="Random Forest Prediction", linestyle="--")
plt.plot(y_pred_lr, label="Linear Regression Prediction", linestyle="--")
plt.plot(y_pred_lstm, label="LSTM Prediction", linestyle="--")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.title("Electric Car Sales Prediction Using ML Models")
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.6)
plt.scatter(y_test, y_pred_knn, label="KNN", alpha=0.6)
plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.6)
plt.scatter(y_test, y_pred_lstm, label="LSTM", alpha=0.6)
plt.plot(y_test, y_test, "r-", label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Electric Car Prices")
plt.legend()
plt.show()
