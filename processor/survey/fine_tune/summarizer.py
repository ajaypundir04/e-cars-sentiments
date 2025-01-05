import logging
from transformers import pipeline
from utils.utils import Utils
from utils.log_utils import LoggerManager


class Summarizer:
    def __init__(self, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_passages(self, file_path):
        self.logger.info(f"Summarizing passages from file: {file_path}")
        passages = Utils.scrape_data_from_file(file_path)
        combined_passage = " ".join(passages)
        summary = self.summarizer(combined_passage, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        self.logger.info("Passages summarized successfully.")
        return summary
