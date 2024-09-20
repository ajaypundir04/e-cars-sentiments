import requests
from bs4 import BeautifulSoup
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from urllib.parse import urljoin
import logging
import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from utils.log_utils import LoggerManager

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the Jieba tokenizer
jieba.initialize()

# Load Hugging Face NER model and tokenizer
NER_MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"  # You can choose other models depending on language
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

class Utils:
    # Initialize the logger for the entire class
    logger = LoggerManager(logging.INFO).get_logger(__name__)

    @staticmethod
    def preprocess_texts(posts, language='english', use_stemming=False, use_lemmatization=True):
        """
        Preprocess the text data by performing tokenization, removing stopwords,
        converting text to lowercase, and optionally applying stemming or lemmatization.

        Args:
            posts (list): A list of text posts to preprocess.
            language (str): The language of the text (e.g., 'english', 'german', 'chinese').
            use_stemming (bool): Whether to apply stemming to the tokens.
            use_lemmatization (bool): Whether to apply lemmatization to the tokens.

        Returns:
            list: A list of cleaned text posts.
        """
        stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        cleaned_posts = []

        for post in posts:
            # Convert to lowercase
            post = post.lower()

            if language == 'chinese':
                # Tokenize using jieba for Chinese
                tokens = jieba.cut(post)
            else:
                # Tokenize using nltk for other languages
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

    @staticmethod
    def extract_entities(text):
        """
        Extract named entities from the text using the Hugging Face transformers NER pipeline.

        Args:
            text (str): The text to process for entity extraction.

        Returns:
            dict: A dictionary of entities with their labels.
        """
        entities = ner_pipeline(text)
        
        entity_dict = {}
        for entity in entities:
            entity_text = entity['word']
            entity_label = entity['entity']
            entity_dict[entity_text] = entity_label
        
        if not entity_dict:
            Utils.logger.info(f"No entities found in the text: {text}")
        
        return entity_dict

    @staticmethod
    def analyze_sentiment(text, positive, neutral, negative):
        """
        Analyze the sentiment of the text using TextBlob, focusing on electric car-related keywords.
        Identify positive, negative, and neutral factors for EV usage.

        Args:
            text (str): The text to analyze.

        Returns:
            tuple: 
                - sentiment_label (int): 1 for positive, 0 for neutral, -1 for negative
                - sentiment_word (str): 'Positive', 'Neutral', or 'Negative'
                - positive_factors (list): List of positive words found
                - negative_factors (list): List of negative words found
                - neutral_factors (list): List of neutral words found
        """
        words = text.split()
        positive_factors = [word for word in words if word.lower() in positive]
        negative_factors = [word for word in words if word.lower() in negative]
        neutral_factors = [word for word in words if word.lower() in neutral]

        if positive_factors and not negative_factors:
            return 1, 'Positive', positive_factors, negative_factors, neutral_factors
        elif negative_factors and not positive_factors:
            return -1, 'Negative', positive_factors, negative_factors, neutral_factors
        elif positive_factors and negative_factors:
            filtered_text = ' '.join(positive_factors + negative_factors)
            blob = TextBlob(filtered_text)
            sentiment_score = blob.sentiment.polarity

            if sentiment_score > 0:
                return 1, 'Positive', positive_factors, negative_factors, neutral_factors
            elif sentiment_score < 0:
                return -1, 'Negative', positive_factors, negative_factors, neutral_factors
            else:
                return 0, 'Neutral', positive_factors, negative_factors, neutral_factors
        else:
            return 0, 'Neutral', positive_factors, negative_factors, neutral_factors

    @staticmethod
    def scrape_data_without_user(url, tag='div', class_name=None, attr_name=None):
        search_url = f"{url}"
        response = requests.get(search_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            if class_name:
                elements = soup.find_all(tag, class_=class_name)
            elif attr_name:
                elements = soup.find_all(tag, attrs={attr_name: True})
            else:
                elements = soup.find_all(tag)
            
            posts = [element.get_text(strip=True) for element in elements]
            return posts
        else:
            Utils.logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            return []

    @staticmethod
    def scrape_data_from_file(file_path, column_name='Review_Text'):
        """
        Reads data from a .md, .txt, or .csv file and processes it into a list of text elements.
        
        Args:
            file_path (str): The path to the .md, .txt, or .csv file.
            column_name (str): The name of the column to extract data from if the file is a CSV.

        Returns:
            list: A list of text elements from the file.
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
            Utils.logger.error(f"Failed to open file: {file_path}")
            return []

    @staticmethod
    def aggregate_matrix_sentiments(sentiment_matrix):
        """
        Aggregate sentiment counts from a sentiment matrix.

        Args:
            sentiment_matrix (pd.DataFrame): DataFrame containing sentiment classifications for each survey response.

        Returns:
            tuple: Aggregated counts of positive, negative, and neutral sentiments.
        """
        positive_count = (sentiment_matrix == 1).sum().sum()
        negative_count = (sentiment_matrix == -1).sum().sum()
        neutral_count = (sentiment_matrix == 0).sum().sum()

        return positive_count, negative_count, neutral_count
    
    @staticmethod
    def scrape_data_without_user(url, tag='div', class_name=None, attr_name=None):
        """
        """
        search_url = f"{url}"
        response = requests.get(search_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            if class_name:
                elements = soup.find_all(tag, class_=class_name)
            elif attr_name:
                elements = soup.find_all(tag, attrs={attr_name: True})
            else:
                elements = soup.find_all(tag)
            
            posts = [element.get_text(strip=True) for element in elements]
            return posts
        else:
            Utils.logger.error(f"Failed to retrieve data. Status code: {response.status_code}")
            return []

    @staticmethod
    def scrape_data_without_user_with_seed_url(seed_url, keyword, tag='div', class_name=None, attr_name=None, max_depth=1):
        """
        Scrapes data from a public website based on the provided seed URL, keyword, tag, and class/attribute.
        The function follows links from the seed URL to scrape additional pages up to a specified depth.
        
        Args:
            seed_url (str): The initial URL to start scraping from.
            keyword (str): Keyword for filtering content (not used in this function but can be applied if needed).
            tag (str): The HTML tag to search for (e.g., 'div', 'article'). Default is 'div'.
            class_name (str, optional): The class attribute to filter by. Default is None.
            attr_name (str, optional): The attribute name to filter by. Default is None.
            max_depth (int): Maximum depth to follow links. Default is 1.

        Returns:
            list: A list of extracted text content.
        """
        def get_page_content(url):
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                if class_name:
                    elements = soup.find_all(tag, class_=class_name)
                elif attr_name:
                    elements = soup.find_all(tag, attrs={attr_name: True})
                else:
                    elements = soup.find_all(tag)
                return [element.get_text(strip=True) for element in elements], soup
            else:
                Utils.logger.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                return [], None

        all_posts = []
        urls_to_scrape = [seed_url]
        scraped_urls = set()

        for depth in range(max_depth):
            new_urls = []
            for url in urls_to_scrape:
                if url not in scraped_urls:
                    posts, soup = get_page_content(url)
                    all_posts.extend(posts)
                    scraped_urls.add(url)
                    if soup and depth < max_depth - 1:
                        links = soup.find_all('a', href=True)
                        for link in links:
                            new_url = urljoin(url, link['href'])
                            if new_url not in scraped_urls:
                                new_urls.append(new_url)
            urls_to_scrape = new_urls

        return all_posts
