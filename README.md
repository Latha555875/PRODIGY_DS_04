# Twitter Data Analysis and Sentiment Classification

This README provides an overview of the steps taken to analyze Twitter data, including data cleaning, text preprocessing, and sentiment analysis using the TextBlob library. The guide also includes visualizations to help understand the sentiment distribution within the dataset.
## Table of Contents

  1.Setup
  2.Data Loading
  3.Data Cleaning
  4.Text Preprocessing
  5.Sentiment Analysis
  6.Visualizations

**Setup**

Ensure you have the required libraries installed. You can install them using pip:

```sh
pip install pandas textblob matplotlib seaborn
```
**Data Loading**

Load the dataset from a CSV file. Ensure the file path is correct.

```sh
import pandas as pd

x = "/home/pulicherla/Downloads/twitter_additional_data.csv"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
df = pd.read_csv(x, encoding="latin1")
print(df.head())
print(df.columns)
print(df.isnull().sum())
```

**Data Cleaning**

Define a function to clean the tweets by removing URLs, user references, hashtags, punctuations, numbers, special characters, and extra whitespaces. Convert the text to lowercase.

```sh
import re
import string

# Define a function to clean tweets
def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#'
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    # Remove special characters
    tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet)
    # Remove extra whitespaces
    tweet = tweet.strip()
    # Convert text to lowercase
    tweet = tweet.lower()
    return tweet

# Apply the clean_tweet function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_tweet)

# Display the first few rows of the cleaned dataset
print(df[['text', 'cleaned_text']].head())
```

## Text Preprocessing

Preprocess the text data by applying the clean_tweet function to the 'text' column to get a cleaned version of the tweets.
## Sentiment Analysis

Perform sentiment analysis on the cleaned tweets using the TextBlob library. Classify the sentiment of each tweet as Positive, Negative, or Neutral.

```sh
from textblob import TextBlob

# Function to classify sentiment
def classify_sentiment(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['Sentiment'] = df['text'].apply(classify_sentiment)

# Display the first few rows with sentiment
print(df[['text', 'Sentiment']].head())
```

## Visualizations

Visualize the distribution of sentiments within the dataset using a count plot.

```sh
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of sentiments
sns.countplot(x="Sentiment", data=df, palette="dark", hue="Sentiment")
plt.title('Sentiment Distribution')
plt.show()
```

**Conclusion**

This guide outlines the key steps for loading, cleaning, analyzing, and visualizing Twitter data. By using text preprocessing techniques and sentiment analysis with TextBlob, we can classify the sentiment of tweets and understand the overall sentiment distribution within the dataset.
















