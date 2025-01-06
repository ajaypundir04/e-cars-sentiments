import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer,T5ForConditionalGeneration
from utils.log_utils import LoggerManager
import evaluate
from datasets import load_dataset
import torch

from processor.survey.fine_tune.data_loader import DataLoader
from processor.survey.fine_tune.data_preprocessor import DataPreprocessor
from processor.survey.fine_tune.model_trainer import ModelTrainer


class FineTuningWorkflow:
    def __init__(self, log_level=logging.INFO):
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.trained_model_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'
        self.trained_tokenizer_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'

    def execute(self):
        self.logger.info("Starting fine-tuning workflow...")

        # Load dataset
        loader = DataLoader()
        data = loader.load_data()

        self.model = T5ForConditionalGeneration.from_pretrained(self.trained_model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.trained_tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Tokenize dataset
        #tokenizer = AutoTokenizer.from_pretrained("allenai/t5-small-squad2-question-generation")
        preprocessor = DataPreprocessor(self.tokenizer)
        tokenized_dataset = preprocessor.tokenize_dataset(data)

        # Load model and train
        #model = AutoModelForSeq2SeqLM.from_pretrained("allenai/t5-small-squad2-question-generation")
        trainer = ModelTrainer(self.model, self.tokenizer)
        trainer.train_model(tokenized_dataset)
        #self.evaluate_model(loader)

        self.logger.info("Fine-tuning workflow completed successfully.")

    def evaluate_model(self, eval_file="valid.jsonl"):
        eval_dataset = load_dataset("json", data_files={"validation": eval_file})["validation"]
        rouge = evaluate.load("rouge")

        for example in eval_dataset:
            passage = example["context"]
            true_question = example["question"]
            generated_question = self.generate_question(passage)

            print(f"Passage: {passage}")
            print(f"True Question: {true_question}")
            print(f"Generated Question: {generated_question}")
            print("-" * 50)

            rouge.add(prediction=generated_question, reference=true_question)

        # Compute evaluation metrics
        results = rouge.compute()
        print(f"Evaluation Results: {results}")

if __name__ == "__main__":
    workflow = FineTuningWorkflow()
    workflow.execute()
