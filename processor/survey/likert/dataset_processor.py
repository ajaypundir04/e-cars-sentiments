from datasets import Dataset
from transformers import AutoTokenizer

class DatasetProcessor:
    def __init__(self, tokenizer_name="julian-schelb/rup-answer-option-likert-scale"):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_and_prepare_data(self, data):
        """
        Load data into HuggingFace dataset and tokenize it.
        The data should be a dictionary with 'text' and 'label' keys.
        """
        # Convert data into a HuggingFace Dataset object
        dataset = Dataset.from_dict(data)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        
        # Apply tokenizer to the dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets

    def split_data(self, tokenized_datasets, test_size=0.1):
        """Split dataset into training and validation sets"""
        return tokenized_datasets.train_test_split(test_size=test_size)
