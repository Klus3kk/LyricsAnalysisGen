import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sentiment(lyrics):
    return TextBlob(lyrics).sentiment.polarity

def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], bins=20, kde=True, color="skyblue", edgecolor="black")
    plt.title("Sentiment Polarity Distribution of Portishead's Lyrics")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Frequency")
    plt.savefig("results/sentiment_distribution.png")
    plt.show()

def sentiment_analysis_pipeline(file_path):
    df = pd.read_csv(file_path)
    df['sentiment'] = df['cleaned_lyrics'].apply(analyze_sentiment)
    plot_sentiment_distribution(df)
