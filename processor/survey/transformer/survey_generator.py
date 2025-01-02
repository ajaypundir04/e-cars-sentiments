from transformers import pipeline
import logging
from utils.log_utils import LoggerManager  


class SurveyGenerator:
    def __init__(self, data_reader, feature_extractor, question_generator, log_level=logging.INFO):
        """
        Initializes the SurveyQuestionGenerator class.

        Args:
            data_reader (DataReader): Instance of the DataReader class.
            feature_extractor (FeatureExtractor): Instance of the FeatureExtractor class.
            question_generator (LikertQuestionGenerator): Instance of the LikertQuestionGenerator class.
            log_level (int): Logging level for the SurveyQuestionGenerator class (default: logging.INFO).
        """
        # Initialize logging specifically for this class
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        
        # Initialize the components
        self.data_reader = data_reader
        self.feature_extractor = feature_extractor
        self.question_generator = question_generator

    def generate_survey_questions(self, file_paths=None, url=None, column_name="Review_Text", model_name="bert-base-uncased", question_model="bert-base-uncased"):
        """
        Generate survey questions based on the data from file or URL.

        Args:
            file_path (str): Path to the file (.csv, .txt, or .md).
            url (str): URL to fetch the HTML content.
            column_name (str): Column name to extract text from if the file is a CSV.
            model_name (str): Model name for feature extraction.
            question_model (str): Model name for question generation.

        Returns:
            list: Generated Likert scale questions.
        """
        # Log start of the process
        self.logger.info("Starting survey question generation.")

        # Step 1: Read Data
        texts, tokens = self.data_reader.read_data(file_paths=file_paths, url=url, column_name=column_name)

        # Step 2: Extract Features
        self.logger.info(f"Extracting features using model: {model_name}.")
        features = self.feature_extractor.extract_features(texts)

        self.logger.info(f"features ::::::: ${features}")
        # Step 3: Generate Likert Questions
        self.logger.info(f"Generating Likert questions using model: {question_model}.")
        likert_questions = self.question_generator.generate_likert_questions(features)

        # Log completion
        self.logger.info(f"Generated {len(likert_questions)} Likert scale questions.")

        return likert_questions

