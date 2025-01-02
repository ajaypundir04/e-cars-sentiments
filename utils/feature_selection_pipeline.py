from transformers import pipeline
from utils.log_utils import LoggerManager
import jieba
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from flair.models import SequenceTagger
from flair.data import Sentence
import logging

class FeatureSelector:
    # Initialize the logger for the entire class
    logger = LoggerManager(logging.INFO).get_logger(__name__)

    # Initialize the feature extraction pipeline
    checkpoint = "facebook/bart-base"
    feature_extractor = pipeline("feature-extraction", framework="pt", model=checkpoint)

    # Initialize the Flair NER tagger
    ner_model = "flair/ner-english"
    tagger = SequenceTagger.load(ner_model)

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
    def extract_features_transformer(texts):
        """
        Extract contextual embeddings for each text using a transformer model.

        Args:
            texts (list): List of preprocessed text posts.

        Returns:
            dict: A dictionary with each post and its corresponding 768-dimensional feature vector.
        """
        embeddings = {}
        for idx, text in enumerate(texts):
            features = FeatureSelector.feature_extractor(text, return_tensors="pt")[0].numpy()
            reduced_features = features.mean(axis=0)  # Reduce to 768-dimensional vector
            embeddings[f"Post_{idx + 1}"] = reduced_features.tolist()

        return embeddings

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
        FeatureSelector.tagger.predict(sentence)

        entities = {}
        for entity in sentence.get_spans('ner'):
            entity_type = entity.get_label("ner").value
            entity_text = entity.text
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)

        return entities
