from prediction.prediction import predict_future_sales
from processor.sentiment_comparator import SentimentComparator
from processor.processor import process_data_with_url_keyword, process_data_with_url_keyword_from_file
from output.output import plot_sentiment_analysis_with_words
from survey.survey_start import survey_init
from utils.utils import process_and_aggregate_data

import configparser
import sys

def get_keywords():
    # Initialize the parser
    config = configparser.ConfigParser()

    # Read the config file
    config.read('config.ini')

    # Read the language from the command-line arguments
    if len(sys.argv) > 1:
        language = sys.argv[1]
    else:
        raise ValueError("Please provide a language code as a command-line argument (e.g., 'EN', 'DE').")

    # Validate the language
    if language not in config:
        raise ValueError(f"Language '{language}' is not supported in the config file.")

    # Access the keywords and stop_word_language
    positive_keywords = config[language]['positive_keywords'].split(', ')
    negative_keywords = config[language]['negative_keywords'].split(', ')
    neutral_keywords = config[language]['neutral_keywords'].split(', ')
    stop_word_language = config[language]['stop_word_language']

    # Construct the result as a dictionary (map)
    result = {
        "language": stop_word_language,
        "positive": positive_keywords,
        "negative": negative_keywords,
        "neutral": neutral_keywords
    }

    return result




def main():
    
    
    #English
    urls_en = [
        'https://climate.ec.europa.eu/news-your-voice/news/5-things-you-should-know-about-electric-cars-2024-05-14_en',
        'https://www.politico.eu/article/death-of-das-auto-electric-vehicles-germany',
        'https://autovista24.autovistagroup.com/news/german-new-car-market-trouble',
        'https://www.spiegel.de/international/business/electric-shock-an-existential-crisis-in-the-german-auto-industry-a-266bd037-b63a-4c9b-97b5-423866d7080f',
        'https://eurocities.eu/stories/e-car-lessons-from-norway',
        'https://edition.cnn.com/2024/04/24/business/china-ev-industry-competition-analysis-intl-hnk/index.html'
               
               
               ]
    #German
    urls_de = ['https://www.spiegel.de/auto/elektroauto-als-dienstwagen-robert-habeck-will-teure-e-autos-ueber-die-steuer-noch-staerker-foerdern-a-3381ffcc-6544-41ea-8e0f-92f43cf98f49',
            'https://www.autobild.de/artikel/mia-electric-das-auto-meines-lebens-22594967.html',
            'https://www.spiegel.de/auto/elektroauto-als-dienstwagen-robert-habeck-will-teure-e-autos-ueber-die-steuer-noch-staerker-foerdern-a-3381ffcc-6544-41ea-8e0f-92f43cf98f49',
            'https://www.spiegel.de/international/world/a-prisoner-of-war-describes-captivity-in-russia-at-night-i-prayed-i-wouldnt-survive-to-the-next-day-a-a2343696-f237-49cd-8c29-9f566b5e775e',  
            'https://www.spiegel.de/wirtschaft/unternehmen/chinesischer-autokonzern-byd-als-sponsor-der-em-wie-spricht-man-das-aus-buett-a-6ed32aab-4495-4aee-9cdc-c8ef7097ad57',
            'https://www.spiegel.de/auto/citroen-e-c3-im-test-mit-diesem-bezahlbaren-elektrokleinwagen-fuehrt-stellantis-vw-vor-a-0207d38d-e024-492d-be6a-27a8f480b8e2',
            'https://www.manager-magazin.de/unternehmen/autoindustrie/volkswagen-baut-bis-2027-elektroauto-fuer-20-000-euro-a-b7a43eb6-c167-486f-9f97-3aa41b60673d',
            'https://www.auto-motor-und-sport.de/elektroauto/dodge-charger-daytona-infos-daten-bilder',
            'https://www.handelsblatt.com/unternehmen/industrie/hohe-rabatte-auf-e-autos-deutsche-autohersteller-geraten-unter-druck/100011438.html'
            ]
    
    # please change me
    urls = urls_de
    #urls = urls_en

    keyword = 'cars'
    key_word_map = get_keywords()

    
    # Process and aggregate data from URLs
    aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors = process_and_aggregate_data(
        urls, process_data_with_url_keyword, keyword, key_word_map
    )

    # Plot aggregated sentiment analysis results from URLs
    plot_sentiment_analysis_with_words(aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, 
                                       aggregated_negative_factors, aggregated_neutral_factors, "Sentiment Analysis with Web Crawler")

    # File paths to analyze
    file_paths = ['stats/ev_china.md','stats/ev_germany.md','stats/ev_norway.md','stats/hybrid_germany.md', 'stats/stats.md']

    # Process and aggregate data from files
    aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, aggregated_negative_factors, aggregated_neutral_factors = process_and_aggregate_data(
        file_paths, process_data_with_url_keyword_from_file, keyword, key_word_map
    )

    # Plot aggregated sentiment analysis results from files
    plot_sentiment_analysis_with_words(aggregated_sentiment_words, aggregated_sentiment_text, aggregated_positive_factors, 
                                       aggregated_negative_factors, aggregated_neutral_factors, "Sentiment Analysis with Files")
    
    

    future_sales = predict_future_sales(file_paths, key_word_map)
    print(f"Predicted future sales: {future_sales}")

    #survey_init();

if __name__ == '__main__':
    main()
