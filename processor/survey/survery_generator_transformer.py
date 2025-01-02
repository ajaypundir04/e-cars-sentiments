from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
from handler.feature_handler_transformer import FeatureHandlerTransformer
from utils.log_utils import LoggerManager
from .abstract_survey_generator import AbstractSurveyGenerator

class SurveyGeneratorTransformer(AbstractSurveyGenerator):
    def __init__(self, log_level=logging.INFO):
        """
        Initializes the SurveyGeneratorTransformer class.

        Args:
            log_level (int): Logging level. Default is logging.INFO.
        """
        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        # Initialize handlers
        self.feature_handler = FeatureHandlerTransformer(log_level=log_level)
        self.logger.info(f'here iam ${self.feature_handler}')

        # Load T5 model and tokenizer for question generation
        self.model_name = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def generate_survey(self, mode, language, keyword, num_features, file_paths=['stats/ev_china.md', 'stats/ev_germany.md'], urls=[]):
        """
        Generates survey questions based on extracted features.

        Args:
            mode (str): The mode for feature extraction ('url', 'file', or 'all').
            language (str): Language code for analysis, e.g., 'EN', 'DE'.
            keyword (str): Keyword to search for in the analysis.
            num_features (int): Number of top features to extract and base the survey on.
            file_paths (list): List of file paths to process in 'file' mode. Default is ['stats/ev_china.md'].
            urls (list): List of URLs to process in 'url' mode.

        Returns:
            dict: A dictionary containing survey questions and extracted features.
        """
        top_features = []

        if mode in ['file', 'all']:
            self.logger.info("Processing features from files...")
            top_features.extend(self.feature_handler.process_from_file(keyword, language, num_features, file_paths))

        if mode in ['url', 'all']:
            self.logger.info("Processing features from URLs...")
            top_features.extend(self.feature_handler.process_from_url(language, keyword, num_features, urls))

        # Deduplicate features
        flat_features = list(set(top_features))

        # Generate survey questions using the refined features
        survey_questions = [self.generate_question_with_transformer(feature) for feature in flat_features]

        return {
            "survey_questions": survey_questions,
            "features": flat_features
        }

    def generate_question_with_transformer(self, feature):
        """
        Generate a survey question using the T5 model.

        Args:
            feature (str): A feature extracted from the data.

        Returns:
            str: A dynamically generated survey question.
        """
        input_text = f"Generate a survey question about the feature: {feature}"

        # Encode the input text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True)

        # Generate the output
        output_ids = self.model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

        # Decode the output to get the question
        question = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return question

    def display_survey(self, survey_questions):
        """
        Display survey questions with Likert scale options.

        Args:
            survey_questions (list): A list of survey questions to display.

        Returns:
            None
        """
        likert_scale = ["Not Important", "Slightly Important", "Moderately Important", "Very Important", "Extremely Important"]

        print("Survey Questions:")
        for i, question in enumerate(survey_questions, start=1):
            print(f"{i}. {question}")
            print("Response options: " + ", ".join(likert_scale))
            print("\n")
