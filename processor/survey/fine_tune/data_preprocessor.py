import logging
from transformers import AutoTokenizer
from utils.log_utils import LoggerManager


class DataPreprocessor:
    def __init__(self, tokenizer, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.tokenizer = tokenizer

    def preprocess_data(self, batch):
        self.logger.info("Preprocessing data batch for question generation...")
        inputs = [f"<answer> {answer} <context> {context}" for answer, context in zip(batch["answer"], batch["context"])]
        targets = batch["question"]

        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = self.tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
        model_inputs["labels"] = labels

        self.logger.info("Data batch preprocessing complete.")
        return model_inputs

    def tokenize_dataset(self, dataset):
        self.logger.info("Tokenizing dataset for question generation...")
        tokenized_dataset = dataset.map(self.preprocess_data, batched=True)
        self.logger.info("Dataset tokenization complete.")
        return tokenized_dataset
