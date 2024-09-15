# e_cars_sentiments

`e_cars_sentiments` is a Python package designed to analyze sentiments related to electric cars. It leverages text mining techniques to perform sentiment analysis on data scraped from online sources, such as news articles, blogs, forums and survey responses. The package supports multiple languages, including English, German, Chinese, and Norwegian.

## Features

- **Data Scraping**: Collect text data from public websites and online sources using a URL or a list of URLs specified in a configuration file.
- **Text Preprocessing**: Clean and preprocess text data using Jieba for Chinese text segmentation and NLTK for stopword removal and tokenization.
- **Sentiment Analysis**: Analyze text for sentiments related to electric cars, identifying positive, negative, and neutral factors.
- **Survey Analysis**: Analyze sentiment from survey responses, classifying questions and responses into positive, negative, and neutral sentiments. Visualize the  overall sentiment distribution with pie charts.
- **Visualization**: Generate visualizations, including pie charts, to represent sentiment analysis results.
- **Multilingual Support**: Analyze sentiment in multiple languages, including English, German, Chinese, and Norwegian.

## Installation

To install this package, clone the repository and install the dependencies using `pip`:

```bash
git clone https://github.com/yourusername/e_cars_sentiments.git
cd e_cars_sentiments
pip install -r requirements.txt
```

## Usage

### Running the Sentiment Analysis

You can run the sentiment analysis by specifying the mode (`url` or `file`) and the language code (`EN`, `DE`, `CN`, or `NO`) as command-line arguments.

#### Example Usage:

**URL Mode (German):**

```bash
python -m executor.executor --mode=url --language=DE
```
### File Mode (English):

```bash
python -m executor.executor --mode=file --language=EN
```

### Command-Line Arguments

- `--mode`: Specify the mode of data input. Use url to scrape data from the web, file to analyze text from a local file, survey to analyze survey responses, or all to run all modes.
- `--language`: Specify the language code (`EN`, `DE`, `CN`, `NO`) for the sentiment analysis.

### Data Sources

- **URL Mode**
The program will scrape data from a list of URLs specified in the `config.ini` file for the selected language. This file contains positive, negative, and neutral keywords, along with URLs to public online sources for each language.

- **Survey Mode** (To do read from database)
The program reads survey responses and associated question texts from the `survey.ini` file. It then analyzes the sentiment of each question and response, classifying them as positive, negative, or neutral. Results are visualized in aggregated pie charts.

## Challenges

### Data Collection

- **Unauthenticated Crawling**: Scraping data from seed URLs may result in too many requests and 429 errors. It’s recommended to implement rate limiting or save the scraped text in files for offline analysis.
- **Keyword Identification**: Identifying positive, negative, and neutral words in multiple languages can be challenging. The list of keywords has been limited to 200 words per sentiment to improve performance.
- **Review Data**: Collecting reviews from sources like Yelp or Google requires saving them in text files for processing due to potential scraping limitations.
- **Survey Data**: Processing survey data can be done using a weighted scoring system and representing it as a matrix (0,1) with respondents and possible answers. These can then be compared for sentiment trends.

    | Question Sentiment | Response Sentiment | Classified Sentiment  |
    |--------------------|--------------------|------------------------|
    | Positive           | Positive           | Positive               |
    | Positive           | Negative           | Negative               |
    | Negative           | Negative           | Positive               |
    | Negative           | Positive           | Negative               |
    | Neutral            | Neutral            | Neutral                |



### Data Analysis

- **Predicting E-Car Adoption/Sales**: We predict electric car adoption and sales by analyzing sentiment data using a variety of machine learning models, including Linear Regression, KNN, Random Forest, and advanced neural networks like LSTM and Transformer models. These models utilize sentiment factors such as positive, negative, and neutral mentions to predict future sales trends, helping to identify key influences on consumer behavior.

- **Comparative Sentiment Analysis**: To compare sentiments across different regions (Norway, Germany, EU), deep learning models like LSTM and transformer-based models like BERT can be utilized. These models capture the sequence and context of text, helping to identify key deciding factors by analyzing sentiment patterns and focusing on important text elements.

    - [How multilingual is Multilingual BERT?](https://arxiv.org/pdf/1906.01502)
    - [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588)
    - [Understanding BERT - Key to Advanced Language Models](https://www.linkedin.com/pulse/understanding-bert-key-advanced-language-models-m-shivanandhan-f6jtc/)
    - [Building an LLM from Scratch (Sebastian Raschka)](https://sebastianraschka.com/)

## Future Work

- Implementing more sophisticated crawling mechanisms with authentication.
- Enhancing the sentiment analysis with more languages and dialects.
- Building and training neural networks for predictive analytics in electric car adoption and market trends.


### Distribution

 - distribution (url, file, surveys)
        
