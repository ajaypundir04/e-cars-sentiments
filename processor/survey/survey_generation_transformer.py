import logging
import json
from processor.survey.question_generator import QuestionGeneration
from utils.log_utils import LoggerManager
from processor.survey.fine_tune.summarizer import Summarizer
from utils.utils import Utils

class TransformerSurveyGenerator:
    def __init__(self, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.summarizer = Summarizer()
        self.qg = QuestionGeneration()

    def generate_contexts(self, mode, file_paths=['stats/article.txt',
                                                                    'stats/article1.txt'], urls=None):
        """
        Generates 10 survey contexts from summarized articles.

        Args:
            mode (str): 'file', 'url', or 'all'.
            file_paths (list): List of file paths for summarization.
            urls (list): List of URLs for web scraping.

        Returns:
            dict: Dictionary containing 10 contexts with their corresponding answers.
        """
        all_contexts = {}
        answer = "Likely"  # Default Likert-scale response

        if mode in ['file', 'all']:
            for file_path in file_paths:
                summary = self.summarizer.summarize_data(file_path)
                
                # Manually split summary into meaningful sentences
                sentences = summary.replace(';', '.').replace('\n', ' ').split('. ')
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # Ensure we get at least 10 sentences
                start_idx = len(all_contexts) + 1 
                for idx, sentence in enumerate(sentences[:10]):  # Limit to 10 contexts
                    all_contexts[str(start_idx + idx)] = {"context": sentence, "answer": answer}
                print(f"Processed file: {file_path}")
                print(all_contexts)  

        
        if mode in ['url', 'all']:
            for url in urls:
                summary = self.summarizer.summarize_data_from_url(url)
                
                sentences = summary.replace(';', '.').replace('\n', ' ').split('. ')
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for idx, sentence in enumerate(sentences[:10]):  # Limit to 10 contexts
                    contexts[str(idx + 1)] = {"context": sentence, "answer": answer}

        return all_contexts

    def save_contexts_to_json(self, contexts, output_file='contexts_answers_1.json'):
        """
        Saves generated contexts to a JSON file.
        """
        with open(output_file, 'w') as f:
            json.dump(contexts, f, indent=4)
        self.logger.info(f"Contexts saved to {output_file}")


    def generate_survey(self, mode, language, keyword, 
        file_paths=['stats/article.txt',
                    'stats/article1.txt', 'stats/article_2.txt'],urls=None):
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
        answer = 'Likely' # As context and answer are generated from the same summarized data
        # Process each file separately if in 'file' or 'all' mode
        if mode == 'file' or mode == 'all':
            for file_path in file_paths:
        # Summarize the cleaned data
                summary = self.summarizer.summarize_data(file_path)
        # Split the summary into sentences based on the period "."
                sentences = summary.split('.')
        for sentence in sentences:
            # Trim any leading or trailing spaces in each sentence
            sentence = sentence.strip() 
            if sentence: # Check if the sentence is not empty
                # Generate the question using each sentence as input
                question_data = self.qg.generate(sentence, answer)
                question = question_data['question']
                # Log the process
                self.logger.info(f"Data summarized successfully from file. {sentence}")
                self.logger.info(f'Question Generated::{question}')
                self.logger.info(f'Answer::{answer}')
                # Append the generated question to the list
                survey_questions.append(question)

        # If the mode is 'url' or 'all', scrape and process URL data
        if mode == 'url' or mode == 'all':
            for url in urls: 
                # Summarize the cleaned data
                summary = self.summarizer.summarize_data_from_url(url)
                summaries.append(summary)
                # Generate the question using the summarized context as input
                question_data = self.qg.generate(summary, answer) # Generate questions using QuestionGeneration
                question = question_data['question']
                self.logger.info(f"Data summarized successfully from url. {summary}")
                self.logger.info(f'question Generated::{question}')
                self.logger.info(f'answer::{answer}')
                survey_questions.append(question)

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
        for i, question in enumerate(survey_questions, start=1):
            print(f"{i}. {question}")
            print("Response options: " + ", ".join(likert_scale))
            print("\n")

    

if __name__ == "__main__":
    generator = TransformerSurveyGenerator()
    contexts = generator.generate_contexts(mode='file', file_paths=['stats/article.txt',
                    'stats/article1.txt', 'stats/article_2.txt'])
    generator.save_contexts_to_json(contexts)
    quiz = generator.qg.generate_questions(generator.qg.load_contexts_answers('contexts_answers_1.json'))
    print(quiz)
