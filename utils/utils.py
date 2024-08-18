import requests
from bs4 import BeautifulSoup
import jieba
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from urllib.parse import urljoin
import logging
from utils.log_utils import LoggerManager

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)

# Initialize the Jieba tokenizer
jieba.initialize()

class Utils:
    # Initialize the logger for the entire class
    logger = LoggerManager(logging.INFO).get_logger(__name__)

    @staticmethod
    def preprocess_texts(posts, language='english'):
        """
        Preprocess the text data by performing tokenization, removing stopwords,
        converting text to lowercase, and joining words back into sentences.

        Args:
            posts (list): A list of text posts to preprocess.
            language (str): The language of the text (e.g., 'english', 'german', 'chinese').

        Returns:
            list: A list of cleaned text posts.
        """
        stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()

        cleaned_posts = []

        for post in posts:
            # Convert to lowercase
            post = post.lower()

            if language == 'chinese':
                # Tokenize using jieba for Chinese
                tokens = jieba.cut(post)
            else:
                # Split into words for other languages
                tokens = post.split()

            # Remove stopwords
            cleaned_tokens = [word for word in tokens if word not in stop_words]
            
            # Join tokens back into a cleaned sentence
            cleaned_post = ' '.join(cleaned_tokens)
            cleaned_posts.append(cleaned_post)

        return cleaned_posts

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
    def scrape_data_without_user(url, keyword, tag='div', class_name=None, attr_name=None):
        """
        Scrapes data from a public website based on the provided URL, keyword, tag, and class/attribute.
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

    @staticmethod
    def scrape_data_from_file(file_path):
        """
        Reads data from a .md or .txt file and processes it into a list of text elements.
        
        Args:
            file_path (str): The path to the .md or .txt file.

        Returns:
            list: A list of text elements from the file.
        """
        try:
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
