import logging
from datasets import load_dataset
from utils.log_utils import LoggerManager


class DataLoader:
    def __init__(self, train_file="train.jsonl", valid_file="valid.jsonl", log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.train_file = train_file
        self.valid_file = valid_file
        self.dataset = None

    def load_data(self):
        self.logger.info("Loading dataset from files...")
        self.dataset = load_dataset("json", data_files={"train": self.train_file, "validation": self.valid_file})
        self.logger.info("Dataset successfully loaded.")
        return self.dataset
