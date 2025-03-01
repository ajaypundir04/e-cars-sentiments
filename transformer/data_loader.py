# data_loader.py
import configparser
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ElectricCarSentimentDataset(Dataset):
    def __init__(self, config_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = self._load_config_data(config_path)
        
    def _load_config_data(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")

        words, labels = [], []
        sentiment_map = {"positive_keywords": 2, "neutral_keywords": 1, "negative_keywords": 0}

        for section in config.sections():
            for sentiment, label in sentiment_map.items():
                if sentiment in config[section]:
                    words_list = config[section][sentiment].split(", ")
                    words.extend(words_list)
                    labels.extend([label] * len(words_list))

        return list(zip(words, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, label = self.data[idx]
        encoding = self.tokenizer(
            word,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
