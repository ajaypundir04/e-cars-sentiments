import logging
from utils.log_utils import LoggerManager
from processor.survey.fine_tune.summarizer import Summarizer  # Importing Summarizer class
from processor.survey.question_generator import QuestionGeneration  # Importing QuestionGeneration class
from utils.utils import Utils  # Importing utility class for data scraping and cleaning

class TransformerSurveyGenerator:
    def __init__(self, log_level=logging.INFO):
        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.summarizer = Summarizer()  # Initialize Summarizer
        self.qg = QuestionGeneration()  # Initialize QuestionGeneration class
    
    def generate_survey(self, mode, language, keyword, 
                        file_paths=['stats/ev_china.md', 'stats/ev_germany.md', 'stats/ev_norway.md', 'stats/hybrid_germany.md', 
                                    'stats/stats.md'], urls=None):
        """
        Generates survey questions based on the summarized context data for each file.
        
        Args:
            mode (str): The mode for feature extraction ('url', 'file', or 'all').
            language (str): Language code for analysis, e.g., 'EN', 'DE'.
            keyword (str): Keyword to search for in the analysis.
            file_paths (list): List of file paths to process in 'file' mode. .
            url (str): URL for web scraping in 'url' mode.
        
        Returns:
            dict: Contains survey questions and features.
        """
        survey_questions = []
        # Generate survey questions based on the summary
        summaries = []
        answer = 'Likely'  # As context and answer are generated from the same summarized data
        # Process each file separately if in 'file' or 'all' mode
        if mode == 'file' or mode == 'all':
            for file_path in file_paths:
                # Summarize the cleaned data
                summary = self.summarizer.summarize_data(file_path)
                # Generate the question using the summarized context as input
                summaries.append(summary)
                question_data = self.qg.generate(summary, answer)  # Generate questions using QuestionGeneration
                question = question_data['question']
                self.logger.info(f"Data summarized successfully from file. {summary}")
                self.logger.info(f'question Generated::{question}')
                self.logger.info(f'answer::{answer}')
                survey_questions.append({
                    'question': question
                })

        # If the mode is 'url' or 'all', scrape and process URL data
        if mode == 'url' or mode == 'all':
            for url in urls:            
                # Summarize the cleaned data
                summary = self.summarizer.summarize_data_from_url(url)
                summaries.append(summary)
                # Generate the question using the summarized context as input
                question_data = self.qg.generate(summary, answer)  # Generate questions using QuestionGeneration
                question = question_data['question']
                self.logger.info(f"Data summarized successfully from file. {summary}")
                self.logger.info(f'question Generated::{question}')
                self.logger.info(f'answer::{answer}')
                survey_questions.append({
                    'question': question
                })

        return {
            "survey_questions": survey_questions
                            }


    def display_survey(self, survey_questions):
        """
        Display survey questions with Likert scale options.

        Args:
            survey_questions (list): A list of survey questions to display.
        
        Returns:
            None
        """
        likert_scale = ["Not Important", "Slightly Important", "Moderately Important", "Very Important", "Extremely Important"]
        
        print("Survey Questions :")
        for i, item in enumerate(survey_questions, start=1):
            print(f"{i}. {item['question']}")
            print("Response options: " + ", ".join(likert_scale))
            print("\n")
