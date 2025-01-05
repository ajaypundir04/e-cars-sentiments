import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.log_utils import LoggerManager

from processor.survey.fine_tune.data_loader import DataLoader
from processor.survey.fine_tune.data_preprocessor import DataPreprocessor
from processor.survey.fine_tune.model_trainer import ModelTrainer


class FineTuningWorkflow:
    def __init__(self, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

    def execute(self):
        self.logger.info("Starting fine-tuning workflow...")

        # Load dataset
        loader = DataLoader()
        data = loader.load_data()

        # Tokenize dataset
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        preprocessor = DataPreprocessor(tokenizer)
        tokenized_dataset = preprocessor.tokenize_dataset(data)

        # Load model and train
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        trainer = ModelTrainer(model, tokenizer)
        trainer.train_model(tokenized_dataset)

        self.logger.info("Fine-tuning workflow completed successfully.")


if __name__ == "__main__":
    workflow = FineTuningWorkflow()
    workflow.execute()
