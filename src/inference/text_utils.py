from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

vader_analyzer = SentimentIntensityAnalyzer()

def is_neutral_with_textblob(text):
    """Use TextBlob to check if the sentence is subjective or objective."""
    return TextBlob(text).sentiment.subjectivity < 0.3 

def is_neutral_with_vader(text):
    """Use VADER to check if sentiment is neutral."""
    score = vader_analyzer.polarity_scores(text)
    return score["neu"] > 0.85

def is_textual_input(text):
    """Check if input is valid text."""
    return isinstance(text, str) and bool(text.strip())

def is_question(text):
    """Check if the text is a question (ends with '?')."""
    return text.strip().endswith("?")
