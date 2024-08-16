import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils.utils import process_and_aggregate_data
from processor.processor import process_data_with_url_keyword_from_with_multi_file

def process_and_prepare_data_for_ml(sources, process_function, keyword, key_word_map):
    """
    Process and aggregate sentiment data, preparing it for machine learning.
    
    Args:
        sources (list): List of URLs or file paths to process.
        process_function (function): Function to process each source.
        keyword (str): The keyword for filtering or highlighting.
        key_word_map (dict): Dictionary containing language and sentiment keywords for analysis.

    Returns:
        DataFrame: A pandas DataFrame containing the processed sentiment features.
    """
    sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors = process_function(
        sources, keyword, key_word_map
    )
    
    # Ensure all lists have the same length
    min_len = min(len(sentiment_words), len(sentiment_text), len(positive_factors), len(negative_factors), len(neutral_factors))
    
    sentiment_words = sentiment_words[:min_len]
    positive_factors = positive_factors[:min_len]
    negative_factors = negative_factors[:min_len]
    neutral_factors = neutral_factors[:min_len]

    # Prepare features
    data = {
        'positive_count': [len(p) for p in positive_factors],
        'negative_count': [len(n) for n in negative_factors],
        'neutral_count': [len(n) for n in neutral_factors],
        'sentiment_score': [1 if w == 'Positive' else (-1 if w == 'Negative' else 0) for w in sentiment_words]
    }
    
    df = pd.DataFrame(data)
    return df

def predict_future_sales(sources, key_word_map, keyword='cars'):
    """
    Predict future sales based on sentiment analysis of sources.

    Args:
        sources (list): List of URLs or file paths to process.
        key_word_map (dict): Dictionary containing language and sentiment keywords for analysis.
        keyword (str): The keyword for filtering or highlighting. Default is 'cars'.

    Returns:
        float: The predicted future sales value.
    """
    # Step 2: Prepare data for ML
    df = process_and_prepare_data_for_ml(sources, process_data_with_url_keyword_from_with_multi_file, keyword, key_word_map)

    # Assuming we have a target variable, e.g., future sales
    # For this example, let's create a dummy target variable
    df['future_sales'] = np.random.randint(100, 200, size=len(df))

    # Step 3: Train the machine learning model
    X = df[['positive_count', 'negative_count', 'neutral_count', 'sentiment_score']]
    y = df['future_sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Make a prediction for the future
    future_sentiment_data = pd.DataFrame({
        'positive_count': [10],  # Example values; these should be based on actual future sentiment analysis
        'negative_count': [3],
        'neutral_count': [5],
        'sentiment_score': [1]
    })

    future_sales_prediction = model.predict(future_sentiment_data)
    return future_sales_prediction[0]