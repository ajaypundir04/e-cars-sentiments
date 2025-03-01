import configparser
import torch
from torch.utils.data import Dataset

class ElectricCarSentimentDataset(Dataset):
    def __init__(self, config_path, tokenizer):
        self.tokenizer = tokenizer
        self.texts, self.labels = self.load_data_from_config(config_path)

    def load_data_from_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')

        texts, labels = [], []
        sentiment_mapping = {"positive_keywords": 2, "negative_keywords": 0, "neutral_keywords": 1}

        for section in config.sections():
            for sentiment, label in sentiment_mapping.items():
                if sentiment in config[section]:
                    words = config[section][sentiment].split(", ")
                    texts.extend(words)
                    labels.extend([label] * len(words))

        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
