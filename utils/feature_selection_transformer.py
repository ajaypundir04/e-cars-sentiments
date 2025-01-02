from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
from collections import Counter
from utils.log_utils import LoggerManager

class FeatureSelectionTransformer:
    # Initialize the logger for the entire class
    logger = LoggerManager(logging.INFO).get_logger(__name__)

    def __init__(self, model_name="t5-small", fine_tuned_model_path=None):
        """
        Initialize the FeatureSelectionTransformer class with a transformer model.

        Args:
            model_name (str): Pretrained model name. Default is 't5-small'.
            fine_tuned_model_path (str): Path to a fine-tuned model. Default is None.
        """
        # Load T5 tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path or model_name)
        self.model.eval()  # Set model to evaluation mode

    def preprocess_texts(self, posts):
        """
        Preprocess text data for analysis.

        Args:
            posts (list): List of raw text strings.

        Returns:
            list: List of cleaned text strings.
        """
        # Lowercase all text and strip unnecessary whitespace
        cleaned_posts = [post.lower().strip() for post in posts]
        return cleaned_posts

    def extract_features_t5(self, texts, top_n=5):
        """
        Extract features using T5 model.

        Args:
            texts (list): List of text snippets.
            top_n (int): Number of top features to extract.

        Returns:
            dict: A dictionary with each post and its corresponding extracted features.
        """
        extracted_features = {}

        for idx, text in enumerate(texts):
            input_text = f"Extract key features: {text}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
                features = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                extracted_features[f"Post_{idx + 1}"] = features.split(", ")[:top_n]

        return extracted_features

    def scrape_data_without_user(self, url, tag='div', class_name=None):
        """
        Scrapes data from a public website based on the provided URL, tag, and class.

        Args:
            url (str): The URL of the website to scrape.
            tag (str): HTML tag to search for. Default is 'div'.
            class_name (str): Specific class name to filter tags. Default is None.

        Returns:
            list: List of extracted text elements.
        """
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            if class_name:
                elements = soup.find_all(tag, class_=class_name)
            else:
                elements = soup.find_all(tag)

            posts = [element.get_text(strip=True) for element in elements]
            return posts
        else:
            self.logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            return []

    def scrape_data_from_file(self, file_path, column_name='Review_Text'):
        """
        Reads data from a file and processes it into a list of text elements.

        Args:
            file_path (str): Path to the file.
            column_name (str): Column name for .csv files. Default is 'Review_Text'.

        Returns:
            list: List of extracted text elements.
        """
        try:
            if file_path.endswith('.csv'):
                elements = []
                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if column_name in row:
                            elements.append(row[column_name].strip())
                return elements
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                if file_path.endswith('.md'):
                    elements = content.split('\n\n')  # Splitting by double newlines for paragraphs
                elif file_path.endswith('.txt'):
                    elements = content.splitlines()  # Splitting by lines

                elements = [element.strip() for element in elements if element.strip()]
                return elements

        except FileNotFoundError:
            self.logger.error(f"Failed to open file: {file_path}")
            return []

    def analyze_sentiment_t5(self, texts):
        """
        Analyze sentiment using T5.

        Args:
            texts (list): List of text strings.

        Returns:
            dict: A dictionary of texts and their sentiment analysis results.
        """
        sentiment_results = {}

        for idx, text in enumerate(texts):
            input_text = f"Analyze sentiment: {text}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
                sentiment = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                sentiment_results[f"Text_{idx + 1}"] = sentiment

        return sentiment_results
