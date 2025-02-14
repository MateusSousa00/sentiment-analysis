from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

vader_analyzer = SentimentIntensityAnalyzer()

def is_neutral_with_textblob(text):
    """Use TextBlob to check if the sentence is subjective or objective."""
    return TextBlob(text).sentiment.subjectivity < 0.3  # Lower values → More Objective (Neutral)

def is_neutral_with_vader(text):
    """Use VADER to check if sentiment is neutral."""
    score = vader_analyzer.polarity_scores(text)
    return score["neu"] > 0.85  # VADER considers neutral if >85% confidence