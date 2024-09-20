import logging
from utils.utils import Utils
from utils.log_utils import LoggerManager
from utils.feature_selection import FeatureSelection  # Assuming the FeatureSelection class is here


class FeatureExtractor:
    def __init__(self, log_level=logging.INFO):
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

    def extract_features_from_url(self, url, key_word_map):
        """
        
        Args:
            url (str): The URL of the web page to scrape.
            key_word_map (dict): Contains language and other configurations for processing.
            
        Returns:
            dict: Extracted features and entities.
        """
        # Scrape data from the URL
        posts = Utils.scrape_data_without_user(url)
        self.logger.info(f"Scraped {len(posts)} posts from the URL: {url}")

        # Preprocess the text
        cleaned_posts = Utils.preprocess_texts(posts, key_word_map['language'])
        self.logger.info(f'Cleaned posts from web: {cleaned_posts}')

        # Extract features using TF-IDF
        features = FeatureSelection.extract_features_tfidf(cleaned_posts)
        self.logger.info(f"Extracted features (keywords) using TF-IDF: {features}")

        # Extract Named Entities
        entities = [FeatureSelection.extract_entities(text) for text in cleaned_posts]
        self.logger.info(f"Extracted named entities from the posts: {entities}")
        
        return {
            "features": features,
            "entities": entities
        }

    def extract_features_from_file(self, file_path, key_word_map):
        """
        Extract features (topics, keywords) from a file by scraping, cleaning, and performing feature extraction.
        
        Args:
            file_path (str): The path to the file (CSV, TXT, or MD).
            key_word_map (dict): Contains language and other configurations for processing.
            
        Returns:
            dict: Extracted features and entities.
        """
        # Scrape data from the file
        posts = Utils.scrape_data_from_file(file_path)
        self.logger.info(f"Scraped {len(posts)} posts from the file: {file_path}")

        # Preprocess the text
        cleaned_posts = Utils.preprocess_texts(posts, key_word_map['language'])
        self.logger.info(f'Cleaned posts from file: {cleaned_posts}')

        # Extract features using TF-IDF
        features = FeatureSelection.extract_features_tfidf(cleaned_posts)
        self.logger.info(f"Extracted features (keywords) using TF-IDF: {features}")

        # Extract Named Entities
        #entities = [FeatureSelection.extract_entities(text) for text in cleaned_posts]
        #self.logger.info(f"Extracted named entities from the file: {entities}")

        return {
            "features": features,
         #   "entities": entities
        }

    def process_and_aggregate_features(self, sources, process_function, key_word_map):
        """
        Process and aggregate features from a list of sources using the specified processing function.
        
        Args:
            sources (list): List of URLs or file paths.
            process_function (function): Function used to process each source.
            key_word_map (dict): Contains language and other configurations for processing.
            
        Returns:
            dict: Aggregated extracted features and entities from all sources.
        """
        aggregated_features = {}
        aggregated_entities = []

        for source in sources:
            result = process_function(source, key_word_map)

            # Aggregate features
            aggregated_features.update(result["features"])

            # Aggregate entities
            #aggregated_entities.extend(result["entities"])

        self.logger.info("Aggregated features from all sources:")
        self.logger.info(f"Features: {aggregated_features}")
        #self.logger.info(f"Entities: {aggregated_entities}")

        return {
            "features": aggregated_features,
            #"entities": aggregated_entities
        }
