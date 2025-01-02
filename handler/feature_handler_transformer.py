from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import configparser
from utils.log_utils import LoggerManager
from processor.feature_extrator_transformer import TransformerFeatureExtractor  # Assuming this is the correct import

class FeatureHandlerTransformer:
    def __init__(self, model_name="t5-small", fine_tuned_model_path=None, log_level=logging.INFO):
        """
        Initialize the FeatureHandler with a transformer model and logging.

        Args:
            model_name (str): Pretrained model name. Default is 't5-small'.
            fine_tuned_model_path (str): Path to the fine-tuned model. Default is None.
            log_level (int): Logging level. Default is logging.INFO.
        """
        # Initialize the logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        # Load tokenizer and model (fine-tuned if specified)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path or model_name)
        self.model.eval()  # Set model to evaluation mode

        # Feature extractor for traditional methods
        self.feature_extractor = TransformerFeatureExtractor(log_level)
        self.logger.info('initialized')

        # Domain-specific configurations
        #self.domain_specific_stopwords = set([
         #   'car', 'electric', 'vehicle', 'long', 'need', 'great', 'good', 'new',
          #  'smooth', 'love', 'nice', 'buy', 'purchase', 'adoption', 'automobile',
          #  'drive', 'use', 'feel'
        #])

        #self.electric_car_keywords = set([
        #    'battery', 'range', 'charging', 'acceleration', 'EV', 'charging station',
        #    'fast charging', 'sustainability', 'eco-friendly', 'emissions', 'autonomous',
        #    'electric motor', 'lithium-ion', 'renewable energy', 'charging infrastructure',
        #    'smart grid', 'battery capacity', 'carbon footprint', 'regenerative braking',
        #    'torque', 'powertrain', 'electric range', 'energy efficiency', 'mileage'
        #])

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
            for batch in data_loader:
                input_ids, target_ids = batch
                input_ids, target_ids = input_ids.squeeze(1), target_ids.squeeze(1)
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        self.model.eval()  # Set model back to evaluation mode

    def extract_features_with_transformer(self, text_snippets):
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

    def process_from_url(self, language, keyword, num_features, urls):
        """
        Process and extract top features from URLs using T5.

        Args:
            language (str): Language for processing.
            keyword (str): Keyword for extraction.
            num_features (int): Number of top features to extract.
            urls (list): List of URLs.

        Returns:
            list: Top N features extracted from URLs.
        """
        # Traditional feature extraction (to get raw text snippets)
        raw_features = self.feature_extractor.extract_features_from_url(urls, keyword, language)
        self.logger.info(f'initialized:::${raw_features}')


        # Use T5 to refine features
        t5_features = self.extract_features_with_transformer(raw_features)
        return t5_features[:num_features]

    def process_from_file(self, keyword, language, num_features, file_paths):
        """
        Process and extract top features from files using T5.

        Args:
            keyword (str): Keyword for feature extraction.
            language (str): Language code to use for analysis.
            num_features (int): Number of top features to extract.
            file_paths (list): List of file paths to extract features from.

        Returns:
            list: Top N features extracted from files.
        """
        # Traditional feature extraction (to get raw text snippets)
        raw_features = self.feature_extractor.extract_features_from_file(file_paths, keyword, language)
        self.logger.info(f'initialized:::file::::::${raw_features}')


        # Use T5 to refine features
        t5_features = self.extract_features_with_transformer(raw_features)
        return t5_features[:num_features]
