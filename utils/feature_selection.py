import requests
from bs4 import BeautifulSoup
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

import concurrent.futures
from flair.models import SequenceTagger
from flair.data import Sentence

import logging
from utils.log_utils import LoggerManager
from urllib.parse import urljoin
from collections import Counter
import csv

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the Jieba tokenizer
jieba.initialize()

# Load Flair's pre-trained NER model
tagger = SequenceTagger.load('ner')

class FeatureSelection:
    # Initialize the logger for the entire class
    logger = LoggerManager(logging.INFO).get_logger(__name__)

    @staticmethod
    def preprocess_texts(posts, language='english', use_stemming=False, use_lemmatization=True):
        """
        Preprocess the text data by performing tokenization, removing stopwords,
        converting text to lowercase, and optionally applying stemming or lemmatization.
        """
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

    @staticmethod
    def extract_features_tfidf(texts, top_n=5):
        """
        Extract features (keywords) using TF-IDF method from a list of texts.

        Args:
            texts (list): List of preprocessed text posts.
            top_n (int): Number of top features to extract per post.

        Returns:
            dict: A dictionary with each post and its corresponding top features.
        """
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        top_features = {}
        dense = tfidf_matrix.todense()

        for idx, text in enumerate(dense.tolist()):
            scores = dict(zip(feature_names, text))
            top_words = Counter(scores).most_common(top_n)
            top_features[f"Post_{idx + 1}"] = top_words

        return top_features

    @staticmethod
    def extract_entities(text):
        """
        Extract Named Entities using Flair's NER model from a given text.

        Args:
            text (str): The text from which to extract entities.

        Returns:
            dict: A dictionary containing entities by their label (e.g., PERSON, ORG).
        """
        sentence = Sentence(text)
        tagger.predict(sentence)

        entities = {}
        for entity in sentence.get_spans('ner'):
            entity_type = entity.get_label("ner").value
            entity_text = entity.text
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)

        return entities

    @staticmethod
    def analyze_sentiment(text):
        """
        Analyze sentiment of the text using TextBlob.

        Args:
            text (str): The text to analyze.

        Returns:
            tuple: The polarity (positive/negative/neutral) and the subjectivity.
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    @staticmethod
    def scrape_data_without_user(url, tag='div', class_name=None):
        """
        Scrapes data from a public website based on the provided URL, tag, and class.
        """
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            if class_name:
                elements = soup.find_all(tag, class_=class_name)
            else:
                elements = soup.find_all(tag)

            posts = [element.get_text(strip=True) for element in elements]
            return posts
        else:
            FeatureSelection.logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            return []

    @staticmethod
    def scrape_data_from_file(file_path, column_name='Review_Text'):
        """
        Reads data from a .md, .txt, or .csv file and processes it into a list of text elements.
        """
        try:
            if file_path.endswith('.csv'):
                elements = []
                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if column_name in row:
                            elements.append(row[column_name].strip())
                return elements
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                if file_path.endswith('.md'):
                    elements = content.split('\n\n')  # Splitting by double newlines for paragraphs
                elif file_path.endswith('.txt'):
                    elements = content.splitlines()  # Splitting by lines

                elements = [element.strip() for element in elements if element.strip()]
                return elements

        except FileNotFoundError:
            FeatureSelection.logger.error(f"Failed to open file: {file_path}")
            return []
