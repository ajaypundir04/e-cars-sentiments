import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import scrape_data_without_user


class SentimentComparator:
    def __init__(self, model_type="LLM", model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialize the SentimentComparator with the chosen model.
        - model_type: "LLM" for using a transformer-based model, "LSTM" for using a traditional neural network.
        - model_name: The pre-trained model name for LLM.
        """
        self.model_type = model_type
        if model_type == "LLM":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif model_type == "LSTM":
            self.tokenizer = Tokenizer(num_words=5000)
            self.model = self._build_lstm_model()

    def _build_lstm_model(self):
        """
        Build a simple LSTM model for sentiment analysis.
        """
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a given text using the chosen model.
        Returns a sentiment score and label.
        """
        if self.model_type == "LLM":
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            scores = outputs.logits.detach().numpy()
            scores = np.squeeze(scores)
            sentiment_score = np.argmax(scores)
            sentiment_labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
            return sentiment_score, sentiment_labels[sentiment_score]

        elif self.model_type == "LSTM":
            self.tokenizer.fit_on_texts([text])
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=100)
            prediction = self.model.predict(padded_sequence)
            sentiment_score = int(prediction.round().item())
            sentiment_labels = ['negative', 'positive']
            return sentiment_score, sentiment_labels[sentiment_score]

    def compare_sentiments(self, sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors):
        """
        Compare sentiments based on the processed sentiment factors.
        Returns a dictionary of sentiments.
        """
        sentiments = {}
        for i, word in enumerate(sentiment_words):
            title = f"Blog {i + 1}"
            content = sentiment_text[i]
            sentiments[title] = {
                'sentiment_word': word,
                'positive_factors': positive_factors[i] if i < len(positive_factors) else None,
                'negative_factors': negative_factors[i] if i < len(negative_factors) else None,
                'neutral_factors': neutral_factors[i] if i < len(neutral_factors) else None
            }
        return sentiments

    def plot_sentiment_comparison(self, sentiments):
        """
        Plot the comparison of sentiments across different blogs.
        sentiments: A dictionary with blog titles as keys and sentiment information as values.
        """
        titles = list(sentiments.keys())
        scores = [1 if 'positive' in sentiments[title]['sentiment_word'].lower() else
                  (-1 if 'negative' in sentiments[title]['sentiment_word'].lower() else 0) for title in titles]
        labels = [sentiments[title]['sentiment_word'] for title in titles]

        plt.figure(figsize=(10, 6))
        plt.barh(titles, scores, color=['green' if 'positive' in label.lower() else 
                                        ('red' if 'negative' in label.lower() else 'blue') for label in labels])
        plt.xlabel('Sentiment Score')
        plt.title('Sentiment Comparison Across Blogs')
        plt.xlim(-2, 2)
        plt.yticks(titles)
        plt.show()

if __name__ == '__main__':
    # Assuming `process_data_with_url_keyword` is defined elsewhere and returns the following:
    # sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors
    
    # Replace the following with the actual call to `process_data_with_url_keyword`
    # Example:
    # sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors = process_data_with_url_keyword(url, keyword)
    sentiment_words = ['Positive', 'Negative', 'Neutral', 'Positive']
    sentiment_text = [
        "Electric cars are fantastic and very efficient.",
        "I'm concerned about the range of electric cars.",
        "Charging infrastructure is still limited in many areas.",
        "The future of cars is electric, and it's bright."
    ]
    positive_factors = [['fantastic', 'efficient'], [], [], ['bright']]
    negative_factors = [[], ['concerned'], ['limited'], []]
    neutral_factors = [[], [], ['infrastructure'], []]

    comparator = SentimentComparator(model_type="LLM")  # or "LSTM"
    sentiments = comparator.compare_sentiments(sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors)
    
    for title, sentiment in sentiments.items():
        print(f"{title}: {sentiment['sentiment_word']} - Positive: {sentiment['positive_factors']}, Negative: {sentiment['negative_factors']}, Neutral: {sentiment['neutral_factors']}")

    comparator.plot_sentiment_comparison(sentiments)
