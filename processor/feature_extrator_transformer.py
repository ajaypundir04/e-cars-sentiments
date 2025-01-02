from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
from utils.log_utils import LoggerManager


class TransformerFeatureExtractor:
    def __init__(self, model_name="t5-small", fine_tuned_model_path=None, log_level=logging.INFO):
        """
        Initialize the TransformerFeatureExtractor with a transformer model and logging.

        Args:
            model_name (str): Pretrained model name. Default is 't5-small'.
            fine_tuned_model_path (str): Path to the fine-tuned model. Default is None.
            log_level (int): Logging level. Default is logging.INFO.
        """
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.logger.info('here iam:::::this is it')    
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path or model_name)
        self.model.eval()  # Set model to evaluation mode

    def fine_tune_model(self, training_data, epochs=3, batch_size=16, learning_rate=5e-5):
        """
        Fine-tune the T5 model on domain-specific data.

        Args:
            training_data (list): List of tuples (input_text, target_text).
            epochs (int): Number of training epochs. Default is 3.
            batch_size (int): Batch size. Default is 16.
            learning_rate (float): Learning rate. Default is 5e-5.
        """
        from torch.utils.data import DataLoader
        from transformers import AdamW

        # Prepare the dataset
        tokenized_data = [
            (
                self.tokenizer.encode(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"),
                self.tokenizer.encode(target_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            )
            for input_text, target_text in training_data
        ]
        data_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for input_ids, target_ids in data_loader:
                input_ids, target_ids = input_ids.squeeze(1), target_ids.squeeze(1)
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        self.model.eval()  # Set model back to evaluation mode

    def extract_features(self, text_snippets):
        """
        Use the T5 model to extract features from text.

        Args:
            text_snippets (list): List of text snippets to process.

        Returns:
            list: Extracted features for each text snippet.
        """
        features = []
        for snippet in text_snippets:
            input_text = f"Extract features: {snippet}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
                feature_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                features.append(feature_text)

        return features

    def process_from_url(self, url, num_features=5):
        """
        Process and extract features from a URL using the T5 model.

        Args:
            url (str): URL to scrape and process.
            num_features (int): Number of top features to extract.

        Returns:
            list: Extracted features.
        """
        from utils.utils import Utils  # Assuming Utils contains scraping logic

        posts = Utils.scrape_data_without_user(url)
        self.logger.info(f"Scraped {len(posts)} posts from the URL: {url}")

        features = self.extract_features(posts)
        return features[:num_features]

    def process_from_file(self, file_path, num_features=5):
        """
        Process and extract features from a file using the T5 model.

        Args:
            file_path (str): File path to read and process.
            num_features (int): Number of top features to extract.

        Returns:
            list: Extracted features.
        """
        from utils.utils import Utils  # Assuming Utils contains file reading logic

        posts = Utils.scrape_data_from_file(file_path)
        self.logger.info(f"Scraped {len(posts)} posts from the file: {file_path}")

        features = self.extract_features(posts)
        return features[:num_features]
