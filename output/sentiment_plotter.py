import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

class SentimentPlotter:
    @staticmethod
    def plot_sentiment_analysis_with_words(sentiment_words, sentiment_text, positive_factors, negative_factors, neutral_factors, plot_title="Sentiment Analysis"):
        """
        Plots the sentiment analysis results in a pie chart, including positive, negative, and neutral factors.
        Also plots a word cloud, a frequency bar chart, and K-means clustering between factors.

        Args:
            sentiment_words (list): The list of sentiment words (e.g., 'Positive', 'Neutral', 'Negative').
            sentiment_text (list): The list of actual texts corresponding to the sentiments.
            positive_factors (list): The list of positive words related to EVs.
            negative_factors (list): The list of negative words related to EVs.
            neutral_factors (list): The list of neutral words related to EVs.
        """
        # Count the occurrences of each sentiment category
        sentiment_counts = {
            'Positive': len(positive_factors),
            'Negative': len(negative_factors),
            'Neutral': len(neutral_factors)
        }
        
        # Check if there is any data to plot
        if not sentiment_counts or all(v == 0 for v in sentiment_counts.values()):
            print("No data to plot.")
            return    

        # Create a word cloud from the sentiment text
        combined_text = ' '.join(sentiment_text)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

        # Calculate word frequencies for the bar chart
        all_factors = positive_factors + negative_factors + neutral_factors
        word_frequencies = Counter(all_factors).most_common(10)  # Get top 10 words

        # K-Means Clustering
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(all_factors)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(X)
        
        # Reduce dimensions for visualization using PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(X.toarray())
        reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

        # Assign colors based on the sentiment
        sentiment_colors = []
        for factor in all_factors:
            if factor in positive_factors:
                sentiment_colors.append('green')
            elif factor in negative_factors:
                sentiment_colors.append('red')
            else:
                sentiment_colors.append('blue')

        # Get the top 3 most frequent positive, negative, and neutral words in each cluster
        clusters = kmeans.labels_
        cluster_top_words = {}

        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_words = [all_factors[idx] for idx in cluster_indices]
            
            pos_words = [word for word in cluster_words if word in positive_factors]
            neg_words = [word for word in cluster_words if word in negative_factors]
            neu_words = [word for word in cluster_words if word in neutral_factors]
            
            top_pos_words = Counter(pos_words).most_common(3)
            top_neg_words = Counter(neg_words).most_common(3)
            top_neu_words = Counter(neu_words).most_common(3)
            
            cluster_top_words[i] = {
                'Positive': top_pos_words,
                'Negative': top_neg_words,
                'Neutral': top_neu_words
            }

        # Plot everything on the same screen using subplots
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(plot_title, fontsize=16)

        # Plot the pie chart for sentiment analysis
        axs[0, 0].pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=['green', 'red', 'blue'])
        axs[0, 0].set_title('Sentiment Analysis Results')

        # Plot the word cloud
        axs[0, 1].imshow(wordcloud, interpolation='bilinear')
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Word Cloud of Sentiment Text')

        # Plot the frequency bar chart
        words, frequencies = zip(*word_frequencies)
        axs[1, 0].barh(words, frequencies, color='skyblue')
        axs[1, 0].set_xlabel('Frequency')
        axs[1, 0].set_title('Top Words by Frequency')
        axs[1, 0].invert_yaxis()  # Highest frequency at the top

        # Plot K-Means Clustering with sentiment colors
        scatter = axs[1, 1].scatter(reduced_features[:, 0], reduced_features[:, 1], c=sentiment_colors, s=100)
        axs[1, 1].scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], c='black', s=200, alpha=0.75, label='Cluster Centers')
        axs[1, 1].set_title('K-Means Clustering of Sentiment Factors')
        axs[1, 1].set_xlabel('PCA Component 1')
        axs[1, 1].set_ylabel('PCA Component 2')

        # Create a box inside the plot to show the top 3 words for each sentiment type
        box_text = ""
        for sentiment, color in [('Positive', 'green'), ('Negative', 'red'), ('Neutral', 'blue')]:
            words = cluster_top_words[0].get(sentiment, [])
            words_text = "\n".join([f"{word}: {count}" for word, count in words])
            box_text += f"{sentiment} Words:\n{words_text}\n\n"

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        axs[1, 1].text(1.05, 0.5, box_text, transform=axs[1, 1].transAxes, fontsize=10, verticalalignment='center', bbox=props)

        # Create a custom legend
        legend_labels = ['Positive', 'Negative', 'Neutral']
        custom_lines = [plt.Line2D([0], [0], color='green', lw=4),
                        plt.Line2D([0], [0], color='red', lw=4),
                        plt.Line2D([0], [0], color='blue', lw=4)]
        axs[1, 1].legend(custom_lines, legend_labels, title="Sentiment")

        # Show the plots
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_survey_sentiment_summary(sentiment_summary):
        """
        Plots the aggregated survey sentiment summary as a single pie chart.

        Args:
            sentiment_summary (pd.DataFrame): A DataFrame containing the sentiment counts for each survey question.
        """
        # Sum the sentiment counts across all questions to get the overall sentiment summary
        aggregated_sentiments = sentiment_summary.sum(axis=1)

        # Define the color mapping for sentiments
        color_map = {
            1: 'green',    # Positive
            -1: 'red',     # Negative
            0: 'blue'      # Neutral
        }

        # Prepare the data for the pie chart
        sizes = aggregated_sentiments.values
        labels = ['Positive', 'Negative', 'Neutral']

        # Ensure the colors match the labels order
        colors = [color_map[1], color_map[-1], color_map[0]]

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax.set_title('Survey Sentiment Analysis')

        # Show the plot
        plt.show()
