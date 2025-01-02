from transformers import AutoModel, AutoTokenizer, pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from utils.log_utils import LoggerManager  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from flair.models import SequenceTagger
from flair.data import Sentence
import jieba
import numpy as np 
import torch 
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

class FeatureExtractor:
    def __init__(self, model_name="bert-base-uncased", log_level=logging.INFO):
        """
        Initializes the FeatureExtractor class with a specified BERT model and logging.
        """
        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        # Load tokenizer and model
        self.logger.info("Initializing tokenizer and model.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Initialize NER model for entity extraction
        self.ner_model = "flair/ner-english"
        self.tagger = SequenceTagger.load(self.ner_model)

        # Initialize Feature Selector for text preprocessing and feature extraction
        self.feature_extractor = pipeline("feature-extraction", framework="pt", model=model_name)

        # Predefined categories for features
        self.categories = [
            "Government Incentives",
            "Charging Infrastructure",
            "Fuel and Electricity Costs",
            "Government Policies and Emissions Standards",
            "EV Market Penetration"
        ]

    def preprocess_texts(self, posts, language='english', use_stemming=False, use_lemmatization=True):
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

    def extract_features(self, texts, top_n=5):
        """
        Extracts features from text data using embeddings and TF-IDF.

        Args:
            texts (list): A list of text data to process.
            top_n (int): Number of clusters and top features to extract (default: 5).

        Returns:
            list: Combined features from transformer-based embeddings, NER, and TF-IDF.
        """
        try:
            self.logger.info("Preprocessing text data.")
            cleaned_texts = self.preprocess_texts(texts)

            # Extract features from transformer embeddings
            self.logger.info("Extracting contextual embeddings for preprocessed text.")
            embeddings = self.feature_extractor(cleaned_texts)

            # Ensure all embeddings are flattened to 1D arrays and have consistent dimensions
            embeddings = [
                np.mean(np.array(embedding), axis=0) if isinstance(embedding, list) else embedding
                for embedding in embeddings
            ]

            embeddings_shapes = [e.shape if hasattr(e, 'shape') else np.array(e).shape for e in embeddings]
            self.logger.debug(f"Embeddings shapes: {embeddings_shapes}")
            
            embeddings = np.vstack(embeddings)

            self.logger.info("Performing KMeans clustering on embeddings.")
            kmeans = KMeans(n_clusters=top_n, random_state=0).fit(embeddings)
            clusters = kmeans.cluster_centers_

            # Extract Named Entities from text using NER
            self.logger.info("Extracting named entities from text data.")
            entities = [self.extract_entities(text) for text in cleaned_texts]
            self.logger.debug(f"Extracted named entities: {entities}")

            # Extract TF-IDF features
            self.logger.info("Extracting TF-IDF keywords from text data.")
            vectorizer = TfidfVectorizer(max_features=top_n)
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            keywords = vectorizer.get_feature_names_out()

            features = list(set(keywords))  # Start with TF-IDF keywords
            for entity_dict in entities:
                for entity_type, entity_list in entity_dict.items():
                    features.extend(entity_list)

            # Extract custom features based on semantic similarity
            custom_features = self.extract_custom_features(cleaned_texts)

            features.extend(custom_features)

            self.logger.info(f"Extracted {len(features)} features.")
            self.logger.info(f"Extracted features: {features}")

            return list(set(features))
        except Exception as e:
            self.logger.error(f"An error occurred during feature extraction: {e}")
            raise

    def extract_custom_features(self, texts):
        """
        Dynamically extracts custom features based on semantic similarity with predefined categories.

        Args:
            texts (list): List of preprocessed text data to process.

        Returns:
            list: List of identified custom features based on semantic similarity.
        """
        custom_features = []
        
        # Convert category names into embeddings
        category_embeddings = self.get_category_embeddings(self.categories)

        # Iterate over each text and compare its similarity to each category
        for text in texts:
            text_embedding = self.get_text_embedding(text)
            
            # Calculate cosine similarity between the text embedding and each category embedding
            similarities = cosine_similarity([text_embedding], category_embeddings)[0]

            # Find the category with the highest similarity
            max_similarity_idx = np.argmax(similarities)
            most_similar_category = self.categories[max_similarity_idx]

            custom_features.append(most_similar_category)

        return custom_features

    def get_category_embeddings(self, categories):
        """
        Generates embeddings for the given categories using the model.

        Args:
            categories (list): List of category names (e.g., "Government Incentives").

        Returns:
            np.ndarray: Array of category embeddings.
        """
        category_embeddings = []
        for category in categories:
            category_embedding = self.get_text_embedding(category)
            category_embeddings.append(category_embedding)
        
        return np.array(category_embeddings)

    def get_text_embedding(self, text):
        """
        Generates an embedding for a given text using the pre-trained model.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            np.ndarray: The embedding for the text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean of all token embeddings
        return embedding

    def extract_entities(self, text):
        """
        Extract Named Entities using Flair's NER model from a given text.

        Args:
            text (str): The text from which to extract entities.

        Returns:
            dict: A dictionary containing entities by their label (e.g., PERSON, ORG).
        """
        sentence = Sentence(text)
        self.tagger.predict(sentence)

        entities = {}
        for entity in sentence.get_spans('ner'):
            entity_type = entity.get_label("ner").value
            entity_text = entity.text
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)

        return entities
