import logging
import configparser
from utils.log_utils import LoggerManager
from processor.feature_extractor import FeatureExtractor  # Assuming the FeatureExtractor class is in this module
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class FeatureHandler:
    def __init__(self, log_level=logging.INFO):
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.feature_extractor = FeatureExtractor(log_level)

        # Refined domain-specific stopwords
        self.domain_specific_stopwords = set([
            'car', 'electric', 'vehicle', 'long', 'need', 'great', 'good', 'new',
            'smooth', 'love', 'nice', 'buy', 'purchase', 'adoption', 'automobile',
            'drive', 'use', 'feel'
        ])

        # Improved domain-specific keywords to prioritize in feature extraction
        self.electric_car_keywords = set([
            'battery', 'range', 'charging', 'acceleration', 'EV', 'charging station',
            'fast charging', 'sustainability', 'eco-friendly', 'emissions', 'autonomous',
            'electric motor', 'lithium-ion', 'renewable energy', 'charging infrastructure',
            'smart grid', 'battery capacity', 'carbon footprint', 'regenerative braking',
            'torque', 'powertrain', 'electric range', 'energy efficiency', 'mileage'
        ])

        # Group similar terms into one feature to reduce redundancy
        self.synonym_mapping = {
            'range': ['mileage', 'distance'],
            'battery': ['battery life', 'battery capacity'],
            'charging': ['charging time', 'charge speed'],
            'acceleration': ['torque', 'speed'],
            'sustainability': ['eco-friendly', 'green'],
            'emissions': ['carbon footprint', 'pollution']
        }

    def get_keywords(self, language):
        """
        Reads the keywords configuration for feature extraction from the config file.
        """
        # Initialize the parser
        config = configparser.ConfigParser()

        # Read the config file
        config.read('config.ini')

        # Validate the language
        if language not in config:
            self.logger.error(f"Language '{language}' is not supported in the config file.")
            raise ValueError(f"Language '{language}' is not supported in the config file.")

        # Access the keywords, stop_word_language, and URLs
        positive_keywords = config[language]['positive_keywords'].split(', ')
        negative_keywords = config[language]['negative_keywords'].split(', ')
        neutral_keywords = config[language]['neutral_keywords'].split(', ')
        stop_word_language = config[language]['stop_word_language']
        urls = config[language]['urls'].split(', ')

        # Construct the result as a dictionary (map)
        result = {
            "language": stop_word_language,
            "positive": positive_keywords,
            "negative": negative_keywords,
            "neutral": neutral_keywords,
            "urls": urls
        }
        self.logger.info(f'Result: {result}')
        return result

    def apply_synonym_mapping(self, feature):
        """
        Maps similar terms to a unified feature name based on synonym mapping.

        Args:
            feature (str): The feature to check for synonyms.

        Returns:
            str: The unified feature name, if a synonym is found; otherwise, the original feature.
        """
        feature_lower = feature.lower()
        for key, synonyms in self.synonym_mapping.items():
            if feature_lower in synonyms or feature_lower == key:
                return key
        return feature

    def filter_features(self, feature):
        """
        Filters out irrelevant, overly generic, or stopword-like features.
        
        Args:
            feature (str): The feature to check.
        
        Returns:
            bool: True if the feature should be kept, False otherwise.
        """
        feature_lower = feature.lower()
        return not (feature_lower.isdigit() or feature_lower in self.domain_specific_stopwords)

    def extract_top_features(self, sources, process_function, keyword, language, num_features):
        """
        Extract features from the given sources and log the top N features by their occurrences.
        
        Args:
            sources (list): List of URLs or file paths.
            process_function (function): Function to extract features from the sources.
            keyword (str): Keyword for extraction.
            language (str): Language for processing.
            num_features (int): Number of top features to select.

        Returns:
            list: Top N features extracted and filtered.
        """
        key_word_map = self.get_keywords(language)

        # Process and extract features from the sources
        print(f'console:::${sources}')
        extracted_data = self.feature_extractor.process_and_aggregate_features(sources, process_function, key_word_map)

        # Flatten the list of features and count occurrences
        feature_counter = {}
        for post_features in extracted_data["features"].values():
            for feature, score in post_features:
                feature = self.apply_synonym_mapping(feature)  # Normalize synonyms
                if self.filter_features(feature):  # Filter out generic or irrelevant features
                    # If the feature is a domain-specific keyword, give it higher priority
                    feature_lower = feature.lower()
                    if feature_lower in self.electric_car_keywords:
                        feature_counter[feature] = feature_counter.get(feature, 0) + 5  # Boost for domain relevance
                    else:
                        feature_counter[feature] = feature_counter.get(feature, 0) + 1

        # Sort features by their occurrence count
        sorted_features = sorted(feature_counter.items(), key=lambda x: x[1], reverse=True)

        # Log the top N features
        self.logger.info(f"Top {num_features} Features with their occurrences:")
        for feature, count in sorted_features[:num_features]:
            self.logger.info(f"Feature: {feature}, Occurrences: {count}")

        # Return the top N features
        return sorted_features[:num_features]    

    def process_from_url(self, language, keyword, num_features):
        """
        Process and log top features from URLs.
        """
        key_word_map = self.get_keywords(language)
        urls = key_word_map['urls']
        return self.extract_top_features(urls, self.feature_extractor.extract_features_from_url, keyword, language, num_features)

    def process_from_file(self, keyword, language, num_features, file_paths):
        """
        Process and log top features from files.

        Args:
            keyword (str): Keyword for feature extraction.
            language (str): Language code to use for analysis.
            num_features (int): Number of top features to extract and log.
            file_paths (list): List of file paths to extract features from. Default is ['stats/ev_china.md'].
        """
        # Call the feature extraction method with the provided or default file paths
        return self.extract_top_features(file_paths, self.feature_extractor.extract_features_from_file, keyword, language, num_features)
