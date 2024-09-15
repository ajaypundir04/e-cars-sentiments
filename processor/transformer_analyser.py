import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from utils.log_utils import LoggerManager

class TransformerSentimentAnalyzer:
    def __init__(self, model_name='bert-base-multilingual-cased', max_len=128, batch_size=16, epochs=3, log_level=logging.INFO):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs

        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        self.logger.info("Initialized TransformerSentimentAnalyzer with model: %s", model_name)

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts.iloc[idx]
            label = self.labels.iloc[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    def prepare_data(self, data):
        """
        Prepares the data for training and validation.
        """
        self.logger.info("Preparing data for training and validation")
        try:
            X_train, X_val, y_train, y_val = train_test_split(data['text'], data['target'], test_size=0.2, random_state=23)
            self.logger.debug("Training data size: %d", len(X_train))
            self.logger.debug("Validation data size: %d", len(X_val))

            train_dataset = self.SentimentDataset(X_train, y_train, self.tokenizer, self.max_len)
            val_dataset = self.SentimentDataset(X_val, y_val, self.tokenizer, self.max_len)

            self.logger.info("Data preparation complete")
            return train_dataset, val_dataset
        except Exception as e:
            self.logger.error("Error during data preparation: %s", e)
            raise

    def train(self, train_dataset, val_dataset):
        """
        Trains the model on the provided datasets.
        """
        self.logger.info("Starting model training")
        try:
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch"
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=lambda p: {
                    'accuracy': (p.predictions.argmax(-1) == p.label_ids).astype(float).mean().item()
                }
            )

            trainer.train()
            self.logger.info("Model training completed")
        except Exception as e:
            self.logger.error("Error during model training: %s", e)
            raise

    def evaluate(self, val_dataset):
        """
        Evaluates the model on the validation dataset.
        """
        self.logger.info("Evaluating the model")
        try:
            trainer = Trainer(model=self.model)
            eval_results = trainer.evaluate(val_dataset=val_dataset)
            self.logger.info("Evaluation results - Accuracy: %.4f", eval_results['eval_accuracy'])
            return eval_results['eval_accuracy']
        except Exception as e:
            self.logger.error("Error during model evaluation: %s", e)
            raise

    def predict(self, texts):
        """
        Predicts the sentiment of a list of texts.
        """
        self.logger.info("Predicting sentiment for input texts")
        
        if isinstance(texts, str):  # If a single string is passed, convert it to a list
            texts = [texts]
        
        sentiments = []
        
        try:
            for text in texts:
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                self.logger.info(f'texts::${text}')

                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                with torch.no_grad():
                    output = self.model(input_ids, attention_mask=attention_mask)
                    probs = torch.nn.functional.softmax(output.logits, dim=-1)
                    prediction = torch.argmax(probs, dim=-1).item()
                    sentiments.append(prediction)

            self.logger.info(f"Predicted sentiments: {sentiments}")
            return sentiments

        except Exception as e:
            self.logger.error("Error during prediction: %s", e)
            raise


    def preprocess_texts(self, texts, language):
        """
        Preprocess the text data according to the specified language.
        This method is useful for handling language-specific preprocessing steps.
        """
        self.logger.info("Preprocessing texts for language: %s", language)

        # Example: Language-specific preprocessing can be added here.
        # For simplicity, this example just returns the input text unchanged.
        if language.lower() in ['de', 'en', 'cn', 'norwegian']:
            # Placeholder for actual preprocessing
            self.logger.debug("Text preprocessing completed")
        else:
            self.logger.warning("Language %s is not explicitly supported. Proceeding with default preprocessing.", language)

        return texts

    @staticmethod
    def supported_languages():
        """
        Returns a list of supported languages by the model.
        """
        return ['DE', 'ENGLISH', 'CHINESE', 'NORWEGIAN']
