import logging
from processor.survey.transformer.data_reader import DataReader
from processor.survey.transformer.feature_extractor import FeatureExtractor
from processor.survey.transformer.likert_question_generator import LikertQuestionGenerator
from processor.survey.transformer.survey_generator import SurveyGenerator


class SurveyQuestionGenerationWorkflow:
    def __init__(self, file_paths=None, url=None, column_name="Review_Text", 
                 model_name="bert-base-uncased", question_model="bert-base-uncased", log_level=logging.INFO):
        """
        Initializes the SurveyQuestionGenerationWorkflow class.
        
        Args:
            file_path (str): Path to the file (.csv, .txt, or .md).
            url (str): URL to fetch the HTML content.
            column_name (str): Column name to extract text from if the file is a CSV.
            model_name (str): Model name for feature extraction.
            question_model (str): Model name for question generation.
            log_level (int): Logging level (default: logging.INFO).
        """
        # Initialize logging for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=log_level)

        # Initialize components
        self.file_paths = file_paths
        self.url = url
        self.column_name = column_name
        self.model_name = model_name
        self.question_model = question_model

        self.data_reader = DataReader(log_level=log_level)
        self.feature_extractor = FeatureExtractor(model_name=model_name)
        self.question_generator = LikertQuestionGenerator(model_name=question_model, log_level=log_level)

        # Initialize SurveyQuestionGenerator
        self.survey_generator = SurveyGenerator(self.data_reader, self.feature_extractor, self.question_generator)

    def execute(self):
        """
        Executes the full survey question generation workflow.
        """
        try:
            self.logger.info("Starting the survey question generation workflow.")

            # Generate survey questions
            questions = self.survey_generator.generate_survey_questions(
                file_paths=self.file_paths, url=self.url, column_name=self.column_name, 
                model_name=self.model_name, question_model=self.question_model)

            # Display Generated Questions
            self.logger.info(f"Generated {len(questions)} survey questions.")
            for idx, question in enumerate(questions, start=1):
                print(f"{idx}. {question}")

        except Exception as e:
            self.logger.error(f"An error occurred during the workflow execution: {e}")
            raise

# Example Usage
if __name__ == "__main__":
    workflow = SurveyQuestionGenerationWorkflow(file_paths = ['stats/ev_china.md', 'stats/ev_germany.md', 'stats/ev_norway.md', 'stats/hybrid_germany.md', 'stats/stats.md', 'stats/reviews.csv'], log_level=logging.DEBUG)
    workflow.execute()
