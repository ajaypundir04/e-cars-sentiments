import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import jieba
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Ensure that the required NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class NeuralNetworkSentimentAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger or self._create_default_logger()

    @staticmethod
    def _create_default_logger():
        logger = logging.getLogger('NeuralNetworkSentimentAnalyzer')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger


        def get_page_content(url):
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                if class_name:
                    elements = soup.find_all(tag, class_=class_name)
                elif attr_name:
                    elements = soup.find_all(tag, attrs={attr_name: True})
                else:
                    elements = soup.find_all(tag)
                return [element.get_text(strip=True) for element in elements], soup
            else:
                logging.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                return [], None

        all_posts = []
        urls_to_scrape = [seed_url]
        scraped_urls = set()

        for depth in range(max_depth):
            new_urls = []
            for url in urls_to_scrape:
                if url not in scraped_urls:
                    posts, soup = get_page_content(url)
                    all_posts.extend(posts)
                    scraped_urls.add(url)
                    if soup and depth < max_depth - 1:
                        links = soup.find_all('a', href=True)
                        for link in links:
                            new_url = urljoin(url, link['href'])
                            if new_url not in scraped_urls:
                                new_urls.append(new_url)
            urls_to_scrape = new_urls

        return all_posts

        stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        cleaned_posts = []

        for post in posts:
            # Convert to lowercase
            post = post.lower()

            if language == 'chinese':
                # Tokenize using jieba for Chinese
                tokens = jieba.cut(post)
            else:
                # Split into words for other languages
                tokens = word_tokenize(post)

            # Remove stopwords
            cleaned_tokens = [word for word in tokens if word not in stop_words]

            # Apply stemming or lemmatization if specified
            if use_stemming:
                cleaned_tokens = [stemmer.stem(word) for word in cleaned_tokens]
            elif use_lemmatization:
                cleaned_tokens = [lemmatizer.lemmatize(word) for word in cleaned_tokens]

            # Join tokens back into a cleaned sentence
            cleaned_post = ' '.join(cleaned_tokens)
            cleaned_posts.append(cleaned_post)

        return cleaned_posts

    def train_sentiment_model(self, data):
        sample_size_per_class = 50000
        positive_sample = data[data['target'] == 4].sample(n=sample_size_per_class, random_state=23)
        negative_sample = data[data['target'] == 0].sample(n=sample_size_per_class, random_state=23)

        balanced_sample = pd.concat([positive_sample, negative_sample])
        
        X_train, X_val, y_train, y_val = train_test_split(balanced_sample['text'], balanced_sample['target'], test_size=0.2, random_state=23)

        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_val_vectorized = vectorizer.transform(X_val)

        X_train_vectorized = X_train_vectorized.todense()
        X_val_vectorized = X_val_vectorized.todense()

        encoder = LabelEncoder()
        y_train_encoded = to_categorical(encoder.fit_transform(y_train))
        y_val_encoded = to_categorical(encoder.transform(y_val))

        model = Sequential()
        model.add(Dense(512, input_shape=(X_train_vectorized.shape[1],), activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train_vectorized, y_train_encoded, epochs=10, batch_size=128,
                            validation_data=(X_val_vectorized, y_val_encoded), verbose=1)

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

        return model, history
