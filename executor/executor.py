import ast
import configparser
import logging
import argparse
import numpy as np
import pandas as pd
from prediction.ml_prediction import ECarSentimentPrediction
from processor.transformer_analyser import TransformerSentimentAnalyzer
from utils.log_utils import LoggerManager
from processor.processor import DataProcessor
from output.sentiment_plotter import SentimentPlotter
from processor.survey.survey_analyser import SurveyAnalyser
from utils.utils import Utils

class SentimentAnalysisApp:
    def __init__(self, log_level=logging.INFO):
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.data_processor = DataProcessor(log_level)
        self.ml_prediction = ECarSentimentPrediction()
        self.transformer_analyzer = TransformerSentimentAnalyzer()


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
    
    def get_survey_data(self, language):
        """
        Reads survey responses, questions, and sentiment keywords from the config file.

        Args:
            language (str): Language section in the config file.

        Returns:
            tuple: survey_responses (list of dicts), questions_texts (dict), sentiment_keywords (dict)
        """
        config = configparser.ConfigParser()
        config.read('survey.ini')

        if language not in config:
            self.logger.error(f"Language '{language}' is not supported in the survey.ini file.")
            raise ValueError(f"Language '{language}' is not supported in the survey.ini file.")

        # Parse survey responses and questions texts
        try:
            survey_responses = ast.literal_eval(config[language]['survey_responses'].strip())
            questions_texts = ast.literal_eval(config[language]['questions_texts'].strip())
            positive_keywords = config[language]['positive_keywords'].split(', ')
            negative_keywords = config[language]['negative_keywords'].split(', ')
            neutral_keywords = config[language]['neutral_keywords'].split(', ')
        except Exception as e:
            self.logger.error(f"Error parsing survey data: {e}")
            raise ValueError(f"Error parsing survey data: {e}")

        survey_data = {
            "positive": positive_keywords,
            "negative": negative_keywords,
            "neutral": neutral_keywords,
            "survey_responses":survey_responses,
            "questions_texts":questions_texts
        }

        return survey_data

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
        return aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors

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
        return aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors


    def process_survey(self, language):
        """
        Process and analyze survey responses.
        """
        key_word_map = self.get_survey_data(language)

        # Read the survey responses and question texts from files
        try:
            survey_responses = key_word_map['survey_responses']
            questions_texts = key_word_map['questions_texts']
        except Exception as e:
            self.logger.error(f"Error reading survey responses or question texts: {e}")
            return

        # Initialize SurveyAnalyser
        survey_analyser = SurveyAnalyser(
            positive_keywords=key_word_map['positive'],
            negative_keywords=key_word_map['negative'],
            neutral_keywords=key_word_map['neutral']
        )

        # Analyze survey
        analysis_results = survey_analyser.analyze_survey({
            "survey_responses":survey_responses, "questions_texts":questions_texts})

        # Output the results
        self.logger.info("Survey Sentiment Analysis Summary:")
        self.logger.info(analysis_results["sentiment_summary"])

        # Plot the survey sentiment summary
        SentimentPlotter.plot_survey_sentiment_summary(analysis_results["sentiment_summary"])
        return analysis_results["sentiment_matrix"]



    def run(self, language, keyword):
        """
        Run sentiment analysis in all modes: URL, File, and Survey.
        """
        all_positive_factors, all_negative_factors, all_neutral_factors = [], [], []

        # Run the URL mode
        self.logger.info("Running sentiment analysis from URLs...")
        pos_factors, neg_factors, neu_factors = self.process_from_url(language, keyword)
        all_positive_factors.extend(pos_factors)
        all_negative_factors.extend(neg_factors)
        all_neutral_factors.extend(neu_factors)
        
        # Define file paths
        file_paths = ['stats/ev_china.md', 'stats/ev_germany.md', 'stats/ev_norway.md', 'stats/hybrid_germany.md', 'stats/stats.md', 'stats/reviews.csv']

        # Run the File mode
        self.logger.info("Running sentiment analysis from files...")
        pos_factors, neg_factors, neu_factors = self.process_from_file(file_paths, keyword, language)
        all_positive_factors.extend(pos_factors)
        all_negative_factors.extend(neg_factors)
        all_neutral_factors.extend(neu_factors)

        # Run Survey analysis
        self.logger.info("Running sentiment analysis from survey responses...")
        
        # Process the survey data and get the sentiment matrix
        sentiment_matrix = self.process_survey(language)

        # Aggregate the sentiment counts from the matrix
        survey_pos_count, survey_neg_count, survey_neu_count = Utils.aggregate_matrix_sentiments(sentiment_matrix)

        # Aggregate all the sentiments (URL, File, Survey)
        agg_pos_factors = len(all_positive_factors) + survey_pos_count
        agg_neg_factors = len(all_negative_factors) + survey_neg_count
        agg_neu_factors = len(all_neutral_factors) + survey_neu_count

        # Prepare data and run the ML model
        sentiment_data = pd.DataFrame({
            'positive_count': [agg_pos_factors],
            'negative_count': [agg_neg_factors],
            'neutral_count': [agg_neu_factors],
            'future_sales': [np.random.randint(0, 100)]  # Replace this with actual target variable if available
        })
        
        X, y = self.ml_prediction.prepare_data(sentiment_data)        
        predictions = self.ml_prediction.run(X,y)
        SentimentPlotter.plot_predictions(predictions)

    def process_transformer(self, language, keyword):
        """
        Process and perform sentiment analysis using Transformer model.
        """
        key_word_map = self.get_keywords(language)

        # Get the URLs from the keyword map
        urls = key_word_map['urls']

        # Process and aggregate data from URLs
        _, aggregated_sentiment_text, _, _, _ = self.data_processor.process_and_aggregate_data(
            urls, self.data_processor.process_data_with_url_keyword, keyword, key_word_map
        )

        # Preprocess the aggregated text based on the language
        preprocessed_texts = self.transformer_analyzer.preprocess_texts(aggregated_sentiment_text, language)

        # Perform sentiment analysis using Transformer on the preprocessed text
        sentiments = self.transformer_analyzer.predict(preprocessed_texts)

        # Count positive, negative, and neutral sentiments
        positive_count = sentiments.count(1)
        negative_count = sentiments.count(0)
        neutral_count = len(sentiments) - positive_count - negative_count  # Assuming binary classification

        # Log and return results
        self.logger.info(f"Positive Sentiment Count: {positive_count}")
        self.logger.info(f"Negative Sentiment Count: {negative_count}")
        self.logger.info(f"Neutral Sentiment Count: {neutral_count}")

        # Plotting the results using the new plot method
        SentimentPlotter.plot_transformer_sentiment_summary(positive_count, negative_count, neutral_count)

        return positive_count, negative_count, neutral_count





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis on URLs, Files, Surveys, and Transformer-based Analysis.")
    
    parser.add_argument('--mode', type=str, choices=['url', 'file', 'survey', 'transformer', 'both', 'all'], default='all',
                        help="Mode to run the analysis: 'url', 'file', 'survey', 'transformer', 'both', or 'all'. Default is 'all'.")
    parser.add_argument('--language', type=str, required=True,
                        help="Language code to use for analysis, e.g., 'EN', 'DE'.")
    parser.add_argument('--keyword', type=str, default='cars',
                        help="Keyword to search for in the analysis. Default is 'cars'.")

    args = parser.parse_args()

    app = SentimentAnalysisApp()

    if args.mode == 'url':
        app.process_from_url(args.language, args.keyword)
    elif args.mode == 'file':
        file_paths = ['stats/ev_china.md','stats/ev_germany.md','stats/ev_norway.md','stats/hybrid_germany.md', 'stats/stats.md', 'stats/reviews.csv']
        app.process_from_file(file_paths, args.keyword, args.language)
    elif args.mode == 'survey':
        app.process_survey(args.language)
    elif args.mode == 'transformer':
        app.process_transformer(args.language, args.keyword)
    else:
        app.run(args.language, args.keyword)