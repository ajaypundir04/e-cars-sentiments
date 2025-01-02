from transformers import AutoTokenizer
import logging
from utils.log_utils import LoggerManager  
from utils.utils import Utils  

class DataReader:
    def __init__(self, log_level=logging.INFO):
        """
        Initializes the DataReader class with logging capabilities.
        
        Args:
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        # Initialize logger
        logger_manager = LoggerManager(log_level)
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.logger.info("DataReader initialized with BERT tokenizer.")

    def read_data(self, file_paths=None, url=None, column_name='Review_Text'):
        """
        Reads text data from multiple files or a URL and tokenizes it using a BERT tokenizer,
        utilizing the Utils class for data processing.

        Args:
            file_paths (list): List of file paths (.csv, .txt, or .md).
            url (str): URL to fetch the HTML content.
            column_name (str): Column name to extract text from if the file is a CSV.

        Returns:
            tuple: A list of texts and their corresponding tokenized representations.
        """
        texts = []
        try:
            # Check if file_paths is provided, and read from each file
            if file_paths:
                for file_path in file_paths:
                    self.logger.info(f"Reading data from file: {file_path}")
                    file_texts = Utils.scrape_data_from_file(file_path, column_name)
                    texts.extend(file_texts)  # Add the extracted texts to the list
                    self.logger.debug(f"Extracted {len(file_texts)} entries from file: {file_path}.")
                    self.logger.debug(f"Extracted texts: {file_texts}")

            # Check if url is provided, and fetch data from the URL
            elif url:
                self.logger.info(f"Fetching data from URL: {url}")
                texts = Utils.scrape_data_without_user(url, tag='p')
                self.logger.debug(f"Extracted {len(texts)} paragraphs from the URL.")
            
            # If neither file_paths nor url is provided, raise an error
            else:
                error_msg = "Either file_paths or url must be provided."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Tokenize the texts using HuggingFace tokenizer
            self.logger.info("Tokenizing text data.")
            tokens = [self.tokenizer.tokenize(text) for text in texts]
            self.logger.debug(f"Tokenized {len(tokens)} texts.")
            self.logger.debug(f"Tokenized texts: {tokens}")

        except Exception as e:
            self.logger.error(f"An error occurred in read_data: {e}")
            raise

        return texts, tokens
