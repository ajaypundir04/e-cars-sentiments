import requests
from bs4 import BeautifulSoup
from utils.utils import preprocess_texts, analyze_sentiment, scrape_data_without_user, scrape_data_from_file, scrape_data_without_user_with_seed_url


def process_data_with_url_keyword(url, keyword, key_word_map):
    """
    Process the data by scraping, cleaning, and performing sentiment analysis.
    """
    #posts = scrape_data_without_user(url, keyword)
    posts = scrape_data_without_user_with_seed_url(url, keyword)
    print('language')
    print(key_word_map['language'])
    cleaned_posts = preprocess_texts(posts, key_word_map['language'])
    print('----------------------------------')
    print(f'from web:::${cleaned_posts}')
    sentiment_words = []
    sentiment_text = []
    positive_factors = []
    negative_factors = []
    neutral_factors = []

    for text in cleaned_posts:
        _,word, positives, negatives, neutrals = analyze_sentiment(text, 
                                                                   key_word_map['positive'],
                                                                    key_word_map['neutral'], 
                                                                    key_word_map['negative'])
        sentiment_words.append(word)
        sentiment_text.append(text)
        positive_factors.extend(positives)
        negative_factors.extend(negatives)
        neutral_factors.extend(neutrals)
    

    return sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors


def process_data_with_url_keyword_from_file(file_path, keyword, key_word_map):
    """
    Process the data by scraping, cleaning, and performing sentiment analysis.
    """
    posts = scrape_data_from_file(file_path)
    cleaned_posts = preprocess_texts(posts, key_word_map['language'])
    print('----------------------------------')
    print(f'from file:::${cleaned_posts}')
    sentiment_words = []
    sentiment_text = []
    positive_factors = []
    negative_factors = []
    neutral_factors = []

    for text in cleaned_posts:
        _,word, positives, negatives, neutrals = analyze_sentiment(text, 
                                                                   key_word_map['positive'],
                                                                    key_word_map['neutral'], 
                                                                    key_word_map['negative'])
        sentiment_words.append(word)
        sentiment_text.append(text)
        positive_factors.extend(positives)
        negative_factors.extend(negatives)
        neutral_factors.extend(neutrals)
    

    return sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors



def process_data_with_url_keyword_from_with_multi_file(file_paths, keyword, key_word_map):
    """
    Process the data from multiple file paths by scraping, cleaning, and performing sentiment analysis.
    
    Args:
        file_paths (list): List of file paths to .md or .txt files.
        keyword (str): The keyword for filtering or highlighting.
        key_word_map (dict): Dictionary containing language and sentiment keywords for analysis.

    Returns:
        tuple: Aggregated sentiment words, sentiment text, positive factors, negative factors, and neutral factors.
    """
    all_sentiment_words = []
    all_sentiment_text = []
    all_positive_factors = []
    all_negative_factors = []
    all_neutral_factors = []

    for file_path in file_paths:
        # Scrape the data from the file
        posts = scrape_data_from_file(file_path)
        cleaned_posts = preprocess_texts(posts, key_word_map['language'])

        # Analyze sentiment for each cleaned post
        sentiment_words = []
        positive_factors = []
        negative_factors = []
        neutral_factors = []
        for text in cleaned_posts:
            _,word, positives, negatives, neutrals = analyze_sentiment(text, 
                                                                     key_word_map['positive'], 
                                                                     key_word_map['neutral'], 
                                                                     key_word_map['negative'])
            sentiment_words.append(word)
            positive_factors.append(positives)
            negative_factors.append(negatives)
            neutral_factors.append(neutrals)
        
        # Aggregate results
        all_sentiment_words.extend(sentiment_words)
        all_sentiment_text.extend(cleaned_posts)
        all_positive_factors.extend(positive_factors)
        all_negative_factors.extend(negative_factors)
        all_neutral_factors.extend(neutral_factors)
    
    return all_sentiment_words, all_sentiment_text, all_positive_factors, all_negative_factors, all_neutral_factors
