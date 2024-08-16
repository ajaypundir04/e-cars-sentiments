import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

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

    # Organize texts by sentiment category for the legend
    sentiment_dict = {
        'Positive': positive_factors,
        'Negative': negative_factors,
        'Neutral': neutral_factors
    }

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

    # Get the top 2 most frequent words in each cluster
    clusters = kmeans.labels_
    cluster_top_words = {}

    for i in range(kmeans.n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_words = [all_factors[idx] for idx in cluster_indices]
        cluster_word_freq = Counter(cluster_words).most_common(2)
        cluster_top_words[i] = cluster_word_freq

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

    # Annotate each cluster with the top 2 most frequent words
    for i, (x, y) in enumerate(reduced_cluster_centers):
        top_words = cluster_top_words[i]
        annotation_text = "\n".join([f"{word}: {count}" for word, count in top_words])
        axs[1, 1].annotate(annotation_text, (x, y), fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.7))

    # Create a custom legend
    legend_labels = ['Positive', 'Negative', 'Neutral']
    custom_lines = [plt.Line2D([0], [0], color='green', lw=4),
                    plt.Line2D([0], [0], color='red', lw=4),
                    plt.Line2D([0], [0], color='blue', lw=4)]
    axs[1, 1].legend(custom_lines, legend_labels, title="Sentiment")

    # Show the plots
    plt.tight_layout()
    plt.show()
