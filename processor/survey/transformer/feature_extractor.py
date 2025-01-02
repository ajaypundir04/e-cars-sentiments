import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import logging
import json
from utils.log_utils import LoggerManager
from flair.models import SequenceTagger
from flair.data import Sentence
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self, model_name="bert-base-uncased", ev_keywords_file="processor/survey/transformer/ev_keywords.json", 
                 train_data_file="processor/survey/transformer/train_data.json", log_level=logging.INFO):
        """
        Initializes the FeatureExtractor class with a specified BERT model and logging.
        """
        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)

        # Load tokenizer and model for fine-tuned BERT
        self.logger.info("Initializing tokenizer and model.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)  # Adjust the num_labels based on your dataset
        self.tagger = SequenceTagger.load("flair/ner-multi")  # Ensure you have this model installed or replace with your own

        # Load EV Keywords from file
        self.ev_keywords = self.load_ev_keywords(ev_keywords_file)

        # Load training data from file
        self.train_texts, self.train_labels = self.load_train_data(train_data_file)

        # Fine-tune the BERT model (call inside the constructor)
        self.fine_tune_bert(self.train_texts, self.train_labels)

    def load_ev_keywords(self, file_path):
        """
        Loads the EV-related keywords from a JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                ev_keywords = json.load(f)
            self.logger.info("EV keywords loaded successfully.")
            return ev_keywords
        except Exception as e:
            self.logger.error(f"Error loading EV keywords: {e}")
            raise

    def load_train_data(self, file_path):
        """
        Loads training data (texts and labels) from a JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                train_data = json.load(f)
            train_texts = train_data['train_texts']
            train_labels = train_data['train_labels']
            self.logger.info("Training data loaded successfully.")
            return train_texts, train_labels
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            raise

    def preprocess_texts(self, posts, language='english', use_stemming=False, use_lemmatization=True):
        """
        Preprocess the text data by performing tokenization, removing stopwords,
        converting text to lowercase, and optionally applying stemming or lemmatization.
        """
        stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        cleaned_posts = []

        for post in posts:
            # Convert to lowercase
            post = post.lower()

            # Split into words
            tokens = word_tokenize(post)

            # Remove stopwords
            cleaned_tokens = [word for word in tokens if word not in stop_words]

            # Apply stemming or lemmatization if specified
            if use_stemming:
                cleaned_tokens = [stemmer.stem(word) for word in cleaned_tokens]
            elif use_lemmatization:
                cleaned_tokens = [lemmatizer.lemmatize(word) for word in cleaned_tokens]

            # Join tokens back into a cleaned sentence
            cleaned_post = ' '.join(cleaned_tokens)
            cleaned_posts.append(cleaned_post)

        return cleaned_posts

    def fine_tune_bert(self, train_texts, train_labels):
        """
        Fine-tune the BERT model for EV-specific feature extraction.
        """
        # Prepare dataset for training
        self.logger.info("Preparing dataset for fine-tuning.")

        # Convert to DataFrame and split into train and validation
        data = {'text': train_texts, 'label': train_labels}
        dataset = Dataset.from_dict(data)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = tokenized_datasets.train_test_split(test_size=0.2).values()

        # Initialize Trainer for fine-tuning
        training_args = TrainingArguments(
            output_dir='./results', 
            num_train_epochs=3, 
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
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
            eval_dataset=val_dataset
        )

        # Fine-tune the model
        trainer.train()

    def extract_features(self, texts, top_n=5):
        """
        Extracts features from text data using embeddings and TF-IDF, excluding named entities.
        """
        try:
            self.logger.info("Preprocessing text data.")
            cleaned_texts = self.preprocess_texts(texts)

            # Use the fine-tuned model for feature extraction
            self.logger.info("Extracting features using fine-tuned model.")
            inputs = self.tokenizer(cleaned_texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Extract Named Entities from text using NER
            self.logger.info("Extracting named entities from text data.")
            entities = [self.extract_entities(text) for text in cleaned_texts]
            self.logger.debug(f"Extracted named entities: {entities}")

            # Remove named entities from feature list
            features = self.extract_keywords_without_entities(cleaned_texts, entities, top_n)

            # Extract custom EV-related features based on keyword matching
            custom_features = self.extract_ev_features(cleaned_texts)

            features.extend(custom_features)

            self.logger.info(f"Extracted {len(features)} features.")
            return list(set(features))
        except Exception as e:
            self.logger.error(f"An error occurred during feature extraction: {e}")
            raise

    def extract_keywords_without_entities(self, cleaned_texts, entities, top_n):
        """
        Removes named entities from the list of extracted keywords and returns only the useful ones.
        """
        # Extract TF-IDF features
        self.logger.info("Extracting TF-IDF keywords from text data.")
        vectorizer = TfidfVectorizer(max_features=top_n)
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        keywords = vectorizer.get_feature_names_out()

        # Flatten the list of named entities and remove them from the keywords
        named_entities = [entity_text for entity_dict in entities for entity_list in entity_dict.values() for entity_text in entity_list]
        named_entities_set = set(named_entities)

        # Filter out named entities from the keywords
        filtered_keywords = [keyword for keyword in keywords if keyword not in named_entities_set]
        
        return filtered_keywords

    def extract_ev_features(self, texts):
        """
        Extracts EV-related features based on predefined categories and keywords.
        """
        ev_features = []
        for text in texts:
            for category, keywords in self.ev_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        ev_features.append(category)
                        break  # Only add the category once for each text
        return ev_features

    def extract_entities(self, text):
        """
        Extract Named Entities using Flair's NER model from a given text.
        """
        sentence = Sentence(text)
        self.tagger.predict(sentence)

        entities = {}
        for entity in sentence.get_spans('ner'):
            entity_type = entity.get_label("ner").value
            entity_text = entity.text
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)

        return entities
