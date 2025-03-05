import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.log_utils import LoggerManager

from processor.survey.fine_tune.summarizer import Summarizer
from processor.survey.fine_tune.question_generator import QuestionGenerator


class QuestionGenerationWorkflow:
    def __init__(self, model_dir="./t5_likert_finetuned_ev", log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.model_dir = model_dir

    def execute(self):
        self.logger.info("Starting question generation workflow...")

        # Load model and tokenizer
        self.logger.info("Loading model and tokenizer...")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        summarizer = Summarizer()
        question_generator = QuestionGenerator(model, tokenizer)

        # Generate questions from sample files
        file_paths = ['stats/ev_china.md', 'stats/ev_germany.md', 'stats/ev_norway.md', 
                      'stats/hybrid_germany.md', 'stats/stats.md']
        for file_path in file_paths:
            passage_summary = summarizer.summarize_data(file_path)
            self.logger.info(f"passage_summary::${passage_summary}")
            question = question_generator.generate_question(passage_summary)
            self.logger.info(f"Generated Likert Question: {question}")

        self.logger.info("Question generation workflow completed successfully.")


if __name__ == "__main__":
    workflow = QuestionGenerationWorkflow()
    workflow.execute()
