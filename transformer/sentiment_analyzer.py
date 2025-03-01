import logging
import torch
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.utils import Utils
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

class TransformerMultilingualSentimentAnalyzer:
    def __init__(self, log_level=logging.INFO):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3).to(self.device)

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Initialized TransformerMultilingualSentimentAnalyzer")

    def fine_tune_model(self, dataset_path):
        """
        Fine-tunes the model on a custom dataset for electric car-related sentiment analysis.
        """
        self.logger.info("Loading dataset for fine-tuning...")
        dataset = ElectricCarSentimentDataset(dataset_path, self.tokenizer)
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_dir="./logs",
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained("./fine_tuned_car_sentiment_model")
        self.tokenizer.save_pretrained("./fine_tuned_car_sentiment_model")
        self.logger.info("Fine-tuning completed! Model saved.")

    def classify_sentiments_from_urls(self, urls):
        """
        Scrapes data from multiple URLs and classifies sentiments for electric car-related words.
        """
        all_word_sentiments = {}
        for url in urls:
            self.logger.info(f"Scraping data from {url}")
            texts = Utils.scrape_data_without_user_with_seed_url(url)

            if not texts:
                self.logger.warning(f"No data extracted from {url}")
                continue

            words = [word for text in texts for word in text.split() if len(word) > 2]
            if not words:
                self.logger.warning(f"No significant words found in {url}")
                continue

            word_sentiments = self.classify_sentiments(words)
            self.logger.info("word_sentiments: %s", word_sentiments)
            all_word_sentiments.update(word_sentiments)

        if all_word_sentiments:
            self.plot_sentiment_analysis_with_words(all_word_sentiments)
        else:
            self.logger.warning("No valid data to plot.")
        return all_word_sentiments

    def classify_sentiments(self, words):
        """
        Classifies the sentiment of electric car-related words instead of full lines.
        """
        self.logger.info(f"Processing {len(words)} words for sentiment classification.")
        
        encoded_inputs = self.tokenizer(
            words,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            predictions = torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=-1), dim=-1).cpu().tolist()

        sentiment_labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }

        word_sentiments = {word: sentiment_labels.get(pred, "Unknown") for word, pred in zip(words, predictions)}

        self.logger.info("Completed Word Sentiment Classification")
        return word_sentiments

    def plot_sentiment_analysis_with_words(self, word_sentiments):
        """
        Plots sentiment distribution and generates a word cloud for electric car-related sentiment categories.
        """
        if not word_sentiments:
            self.logger.warning("No sentiment data to plot.")
            return

        sentiment_counts = Counter(word_sentiments.values())

        positive_words = [word for word, sentiment in word_sentiments.items() if sentiment == "Positive"]
        negative_words = [word for word, sentiment in word_sentiments.items() if sentiment == "Negative"]
        neutral_words = [word for word, sentiment in word_sentiments.items() if sentiment == "Neutral"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Pie chart for sentiment distribution
        labels = list(sentiment_counts.keys())
        sizes = list(sentiment_counts.values())
        colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=[colors[label] for label in labels])
        axes[0].set_title('Electric Car Sentiment Distribution')

        # Generate word clouds for each sentiment category
        wordcloud_positive = WordCloud(width=400, height=400, background_color='white').generate(" ".join(positive_words))
        wordcloud_negative = WordCloud(width=400, height=400, background_color='white').generate(" ".join(negative_words))
        wordcloud_neutral = WordCloud(width=400, height=400, background_color='white').generate(" ".join(neutral_words))

        axes[1].imshow(wordcloud_positive, interpolation='bilinear')
        axes[1].axis("off")
        axes[1].set_title("Positive Factors")

        axes[2].imshow(wordcloud_negative, interpolation='bilinear')
        axes[2].axis("off")
        axes[2].set_title("Negative Factors")

        plt.tight_layout()
        plt.show()

class ElectricCarSentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.texts = list(self.data["Word"])
        self.labels = list(self.data["Sentiment"])  # 0: Negative, 1: Neutral, 2: Positive

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
