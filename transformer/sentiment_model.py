import logging
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.utils import Utils
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

class TransformerModel:
    def __init__(self, model_name='bert-base-multilingual-cased', log_level=logging.INFO):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Initialized TransformerModel")

    def fine_tune_model(self, dataset):
        """
        Fine-tunes the model on a dataset.
        """
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_dir="./logs",
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained("./fine_tuned_car_sentiment_model")
        self.tokenizer.save_pretrained("./fine_tuned_car_sentiment_model")
        self.logger.info("Fine-tuning completed! Model saved.")