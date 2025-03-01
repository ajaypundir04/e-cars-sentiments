# model.py
import logging
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from tqdm import tqdm

class TransformerModel:
    def __init__(self, model_name='bert-base-multilingual-cased', log_level=logging.INFO):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Initialized TransformerModel It.")

    def fine_tune_model(self, dataset, batch_size=16, epochs=3, lr=2e-5):
        """
        Fine-tunes the model on a dataset using AdamW optimizer.
        """
        self.logger.info("Starting fine-tuning...")

        train_dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=batch_size
        )

        optimizer = AdamW(self.model.parameters(), lr=lr)

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_input_ids = batch["input_ids"].to(self.device)
                batch_attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                self.model.zero_grad()

                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            avg_loss = total_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")

        # Save the fine-tuned model
        self.model.save_pretrained("./e_car_sentiment_model")
        self.tokenizer.save_pretrained("./e_car_sentiment_model")
        self.logger.info("Fine-tuning completed! Model saved.")
