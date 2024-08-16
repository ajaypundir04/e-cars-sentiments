# e_cars_sentiments

`e_cars_sentiments` is a Python package designed to analyze sentiments related to electric cars. It leverages text mining techniques to perform sentiment analysis on data scraped from online sources, such as news articles, blogs, and forums.

## Features

- **Data Scraping**: Collect text data from public websites and online sources.
- **Text Preprocessing**: Clean and preprocess text data using Jieba and NLTK for better sentiment analysis.
- **Sentiment Analysis**: Analyze text for sentiments related to electric cars, identifying positive, negative, and neutral factors.
- **Visualization**: Generate visualizations, including pie charts, to represent sentiment analysis results.

## Installation

To install this package, clone the repository and install the dependencies using `pip`:

```bash
git clone https://github.com/yourusername/e_cars_sentiments.git
cd e_cars_sentiments
pip install -r requirements.txt
```

To run

```bash
  python -m executor.executor EN
```

## Challenges

### Data Collection
- Unauthenticated Crawling (Download text and save in file)
    - Seed url reults in too many request 429
- Identification of -ve, +ve and neutral words in different language (sentiment_score is not visibile) (limit it 200 words)
  - converted every word in small letter
- unable to read yelp/google reviews (download and save them in text file)
- how to process survey data (weightage based) (put it in 0,1 mtrix with respndents & possible answers) (compare them)

### Data Analysis
- prediction of e-cars adoption/sales  (identify variables) (Nueral Network)

- To compare sentiments across Norway, Germany, and the EU, deep learning models like LSTM and BiLSTM capture the sequence and context of text, while  transformer-based models like BERT and RoBERTa leverage pre-trained knowledge to understand nuanced sentiment. These models help identify key deciding factors by analyzing sentiment patterns and focus on important text elements. (Just a wild thought)
    - https://arxiv.org/pdf/1906.01502 (How multilingual is Multilingual BERT?)
    - https://arxiv.org/abs/1903.09588 (Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence)
    - https://www.linkedin.com/pulse/understanding-bert-key-advanced-language-models-m-shivanandhan-f6jtc/ (BERT stands for Bidirectional Encoder Representations from Transformers)
    - Sebastian Raschka: Buildung an LLM from Scratch (Wishful)