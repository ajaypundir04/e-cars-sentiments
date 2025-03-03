import logging
from transformers import pipeline
from utils.utils import Utils
from utils.log_utils import LoggerManager

class Summarizer:
    def __init__(self, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_data(self, file_path):
        """
        Summarizes text from a given file.
        """
        self.logger.info(f"Summarizing passages from file: {file_path}")
        passages = Utils.scrape_data_from_file(file_path)
        combined_passage = " ".join(passages)
        
        # Increase max_length for a more detailed summary
        summary = self.summarizer(combined_passage, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
        return summary
    
    def summarize_data_from_url(self, url):
        """
        Summarizes text from a given URL.
        """
        self.logger.info(f"Summarizing passages from URL: {url}")
        passages = Utils.scrape_data_without_user_with_seed_url(seed_url=url)
        combined_passage = " ".join(passages)
        
        summary = self.summarizer(combined_passage, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
        return summary
