import configparser
import logging
import argparse
from utils.log_utils import LoggerManager
from processor.processor import DataProcessor
from output.sentiment_plotter import SentimentPlotter
from survey.survey_start import survey_init

class SentimentAnalysisApp:
    def __init__(self, log_level=logging.INFO):
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.data_processor = DataProcessor(log_level)

        
    def get_keywords(self, language):
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
        self.logger.info('----------------------------------------------------')
        self.logger.info(f'Result: {result}')

        return result

    def process_from_url(self, language, keyword):
        """
        Process and aggregate sentiment analysis data from URLs.
        """
        key_word_map = self.get_keywords(language)

        # Get the URLs from the keyword map
        urls = key_word_map['urls']
        
        # Process and aggregate data from URLs
        aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors = self.data_processor.process_and_aggregate_data(
            urls, self.data_processor.process_data_with_url_keyword, keyword, key_word_map
        )

        # Plot aggregated sentiment analysis results from URLs
        SentimentPlotter.plot_sentiment_analysis_with_words(
            aggregated_sentiment_words, aggregated_sentiment_text, 
            aggregated_positive_factors, aggregated_negative_factors, 
            aggregated_neutral_factors, "Sentiment Analysis with Web Crawler"
        )
    
    def process_from_file(self, file_paths, keyword, language):
        """
        Process and aggregate sentiment analysis data from files.
        """
        key_word_map = self.get_keywords(language)

        # Process and aggregate data from files
        aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors = self.data_processor.process_and_aggregate_data(
            file_paths, self.data_processor.process_data_with_url_keyword_from_file, keyword, key_word_map
        )

        # Plot aggregated sentiment analysis results from files
        SentimentPlotter.plot_sentiment_analysis_with_words(
            aggregated_sentiment_words, aggregated_sentiment_text, 
            aggregated_positive_factors, aggregated_negative_factors, 
            aggregated_neutral_factors, "Sentiment Analysis with Files"
        )

    def run(self, language, keyword):
        """
        Run sentiment analysis in both modes: URL and File.
        """
        # Run the URL mode
        self.logger.info("Running sentiment analysis from URLs...")
        self.process_from_url(language, keyword)
        
        # Define file paths
        file_paths = ['stats/ev_china.md','stats/ev_germany.md','stats/ev_norway.md','stats/hybrid_germany.md', 'stats/stats.md']

        # Run the File mode
        self.logger.info("Running sentiment analysis from files...")
        self.process_from_file(file_paths, keyword, language)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis on URLs and/or Files.")
    
    parser.add_argument('--mode', type=str, choices=['url', 'file', 'both'], default='both',
                        help="Mode to run the analysis: 'url', 'file', or 'both'. Default is 'both'.")
    parser.add_argument('--language', type=str, required=True,
                        help="Language code to use for analysis, e.g., 'EN', 'DE'.")
    parser.add_argument('--keyword', type=str, default='cars',
                        help="Keyword to search for in the analysis. Default is 'cars'.")

    args = parser.parse_args()

    app = SentimentAnalysisApp()

    if args.mode == 'url':
        app.process_from_url(args.language, args.keyword)
    elif args.mode == 'file':
        file_paths = ['stats/ev_china.md','stats/ev_germany.md','stats/ev_norway.md','stats/hybrid_germany.md', 'stats/stats.md']
        app.process_from_file(file_paths, args.keyword, args.language)
    else:
        app.run(args.language, args.keyword)
