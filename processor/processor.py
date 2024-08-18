import logging
from utils.utils import Utils
from utils.log_utils import LoggerManager

class DataProcessor:
    def __init__(self, log_level=logging.INFO):
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

    def process_data_with_url_keyword(self, url, keyword, key_word_map):
        """
        Process the data by scraping, cleaning, and performing sentiment analysis.
        """
        posts = Utils.scrape_data_without_user_with_seed_url(url, keyword)
        
        self.logger.info(f"Language: {key_word_map['language']}")
        cleaned_posts = Utils.preprocess_texts(posts, key_word_map['language'])
        
        self.logger.info('----------------------------------')
        self.logger.info(f'From web::: {cleaned_posts}')
        
        sentiment_words = []
        sentiment_text = []
        positive_factors = []
        negative_factors = []
        neutral_factors = []

        for text in cleaned_posts:
            _, word, positives, negatives, neutrals = Utils.analyze_sentiment(
                text, 
                key_word_map['positive'], 
                key_word_map['neutral'], 
                key_word_map['negative']
            )
            sentiment_words.append(word)
            sentiment_text.append(text)
            positive_factors.extend(positives)
            negative_factors.extend(negatives)
            neutral_factors.extend(neutrals)

        return sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors

    def process_data_with_url_keyword_from_file(self, file_path, keyword, key_word_map):
        """
        Process the data by scraping, cleaning, and performing sentiment analysis.
        """
        posts = Utils.scrape_data_from_file(file_path)
        cleaned_posts = Utils.preprocess_texts(posts, key_word_map['language'])
        
        self.logger.info('----------------------------------')
        self.logger.info(f'From file::: {cleaned_posts}')
        
        self.logger.info('-------------Key Word Map---------------------')
        self.logger.info(f'From file::: {key_word_map}')
        
        sentiment_words = []
        sentiment_text = []
        positive_factors = []
        negative_factors = []
        neutral_factors = []

        self.logger.info(f'Positive Keywords::: {key_word_map["positive"]}')
        self.logger.info(f'Negative Keywords::: {key_word_map["negative"]}')

        for text in cleaned_posts:
            _, word, positives, negatives, neutrals = Utils.analyze_sentiment(
                text, 
                key_word_map['positive'], 
                key_word_map['neutral'], 
                key_word_map['negative']
            )
            sentiment_words.append(word)
            sentiment_text.append(text)
            positive_factors.extend(positives)
            negative_factors.extend(negatives)
            neutral_factors.extend(neutrals)

        return sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors

    def process_data_with_url_keyword_from_multi_file(self, file_paths, keyword, key_word_map):
        """
        Process the data from multiple file paths by scraping, cleaning, and performing sentiment analysis.
        
        Args:
            file_paths (list): List of file paths to .md or .txt files.
            keyword (str): The keyword for filtering or highlighting.
            key_word_map (dict): Dictionary containing language and sentiment keywords for analysis.

        Returns:
            tuple: Aggregated sentiment words, sentiment text, positive factors, negative factors, and neutral factors.
        """
        all_sentiment_words = []
        all_sentiment_text = []
        all_positive_factors = []
        all_negative_factors = []
        all_neutral_factors = []

        for file_path in file_paths:
            # Scrape the data from the file
            posts = Utils.scrape_data_from_file(file_path)
            cleaned_posts = Utils.preprocess_texts(posts, key_word_map['language'])

            # Analyze sentiment for each cleaned post
            sentiment_words = []
            positive_factors = []
            negative_factors = []
            neutral_factors = []
            for text in cleaned_posts:
                _, word, positives, negatives, neutrals = Utils.analyze_sentiment(
                    text, 
                    key_word_map['positive'], 
                    key_word_map['neutral'], 
                    key_word_map['negative']
                )
                sentiment_words.append(word)
                positive_factors.append(positives)
                negative_factors.append(negatives)
                neutral_factors.append(neutrals)
            
            # Aggregate results
            all_sentiment_words.extend(sentiment_words)
            all_sentiment_text.extend(cleaned_posts)
            all_positive_factors.extend(positive_factors)
            all_negative_factors.extend(negative_factors)
            all_neutral_factors.extend(neutral_factors)
        
        return all_sentiment_words, all_sentiment_text, all_positive_factors, all_negative_factors, all_neutral_factors

    def process_and_aggregate_data(self, sources, process_function, keyword, key_word_map):
        """
        Process data from a list of sources (URLs or files) using the specified processing function.
        Aggregates the results from all sources.

        Args:
            sources (list): List of URLs or file paths.
            process_function (function): The function to process each source (URL or file).
            keyword (str): The keyword for filtering or highlighting.
            key_word_map (dict): Dictionary containing language and sentiment keywords for analysis.

        Returns:
            tuple: Aggregated sentiment words, sentiment text, positive factors, negative factors, and neutral factors.
        """
        aggregated_sentiment_words = []
        aggregated_sentiment_text = []
        aggregated_positive_factors = []
        aggregated_negative_factors = []
        aggregated_neutral_factors = []

        for source in sources:
            sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors = process_function(source, keyword, key_word_map)

            # Aggregate results
            aggregated_sentiment_words.extend(sentiment_words)
            aggregated_sentiment_text.extend(sentiment_text)
            aggregated_positive_factors.extend(positive_factors)
            aggregated_negative_factors.extend(negative_factors)
            aggregated_neutral_factors.extend(neutral_factors)

        self.logger.info('Aggregated Sentiment Analysis Results:')
        self.logger.info(f'Sentiment Words: {aggregated_sentiment_words}')
        self.logger.info(f'Sentiment Text: {aggregated_sentiment_text}')
        self.logger.info(f'Positive Factors: {aggregated_positive_factors}')
        self.logger.info(f'Negative Factors: {aggregated_negative_factors}')
        self.logger.info(f'Neutral Factors: {aggregated_neutral_factors}')

        return aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors
