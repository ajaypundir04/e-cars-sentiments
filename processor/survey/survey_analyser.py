import logging
import pandas as pd
from utils.log_utils import LoggerManager
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils.utils import Utils

class SurveyAnalyser:
    def __init__(self, positive_keywords, negative_keywords, neutral_keywords, stop_word_language='english', log_level=logging.INFO):
        """
        Initializes the SurveyAnalyser with sentiment classification keywords.

        Args:
            positive_keywords (list): List of keywords indicating positive sentiment.
            negative_keywords (list): List of keywords indicating negative sentiment.
            neutral_keywords (list): List of keywords indicating neutral sentiment.
            stop_word_language (str): The language for stopwords removal (default is English).
        """
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.positive_keywords = [kw.lower() for kw in positive_keywords]
        self.negative_keywords = [kw.lower() for kw in negative_keywords]
        self.neutral_keywords = [kw.lower() for kw in neutral_keywords]
        self.stop_words = set(stopwords.words(stop_word_language))

    def classify_question_sentiment(self, question_text):
        """
        Classifies a survey question based on the defined sentiment keywords.

        Args:
            question_text (str): The survey question text.

        Returns:
            str: The classified sentiment of the question ('Positive', 'Neutral', 'Negative').
        """
        # Convert the question text to lowercase and tokenize
        question_text = question_text.lower()
        tokens = word_tokenize(question_text)

        # Remove stop words from the tokens
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Log filtered tokens for debugging
        #self.logger.info(f"Filtered tokens for question '{question_text}': {filtered_tokens}")

        # Reconstruct the filtered question text
        filtered_text = ' '.join(filtered_tokens)

        # Check for sentiment keywords in the filtered text
        sentiment_label, sentiment_word, _, _, _ = Utils.analyze_sentiment(filtered_text, self.positive_keywords, self.neutral_keywords, self.negative_keywords)
        #self.logger.info(f"Classified Question Sentiment - '{question_text}': {sentiment_word}")
        
        return sentiment_word

    def process_responses(self, survey_responses, questions_texts):
        """
        Processes the survey responses, converting them into a sentiment matrix.

        Args:
            survey_responses (list of dicts): The list of survey responses.
            questions_texts (dict): The dictionary mapping question IDs to their text.

        Returns:
            pd.DataFrame: A DataFrame with the sentiment classification for each response.
        """
        sentiment_matrix = {}

        for question_id, question_text in questions_texts.items():
            question_sentiment = self.classify_question_sentiment(question_text)
            question_responses = []

            for response in survey_responses:
                sentiment_label, _, _, _, _ = Utils.analyze_sentiment(response[question_id], self.positive_keywords, self.neutral_keywords, self.negative_keywords)

                # Adjust the sentiment based on the question's sentiment
                if question_sentiment == 'Positive':
                    # Positive response to positive question is +ve, negative response to positive question is -ve
                    if sentiment_label == 1:
                        final_sentiment = 1
                    elif sentiment_label == -1:
                        final_sentiment = -1
                    else:
                        final_sentiment = 0
                elif question_sentiment == 'Negative':
                    # Negative response to negative question is +ve, positive response to negative question is -ve
                    if sentiment_label == 1:
                        final_sentiment = -1
                    elif sentiment_label == -1:
                        final_sentiment = 1
                    else:
                        final_sentiment = 0
                else:
                    final_sentiment = sentiment_label  # Neutral stays neutral

                # Log the final sentiment of each response
                #self.logger.info(f"Response to '{question_text}' classified as {final_sentiment} (original sentiment: {sentiment_label}): '{response[question_id]}'")

                question_responses.append(final_sentiment)

            sentiment_matrix[question_id] = question_responses

        sentiment_matrix_df = pd.DataFrame(sentiment_matrix)
        self.logger.info(f'Response Sentiment matrix:\n{sentiment_matrix_df}')
        return sentiment_matrix_df

    def summarize_sentiments(self, sentiment_matrix):
        """
        Summarizes the sentiment analysis results.

        Args:
            sentiment_matrix (pd.DataFrame): The DataFrame containing sentiment classifications.

        Returns:
            pd.DataFrame: A summary of the sentiment counts for each question.
        """
        # Count the occurrence of each sentiment for each question
        sentiment_summary = sentiment_matrix.apply(pd.Series.value_counts).fillna(0).astype(int)
        #self.logger.info(f'Sentiment summary:\n{sentiment_summary}')

        return sentiment_summary

    def analyze_survey(self, survey_data):
        """
        Full analysis pipeline: process survey responses and summarize sentiments.

        Args:
            survey_data (dict): The dictionary containing survey responses, questions texts, and sentiment keywords.

        Returns:
            dict: The sentiment matrix and sentiment summary for the survey.
        """
        survey_responses = survey_data["survey_responses"]
        questions_texts = survey_data["questions_texts"]
        
        sentiment_matrix = self.process_responses(survey_responses, questions_texts)
        sentiment_summary = self.summarize_sentiments(sentiment_matrix)
        sentiment_summary.index = ['Negative', 'Neutral', 'Positive']

        return {
            "sentiment_matrix": sentiment_matrix,
            "sentiment_summary": sentiment_summary
        }