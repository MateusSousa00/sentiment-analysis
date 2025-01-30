from transformers import pipeline

# This is good for testing locally
sentiment_model = pipeline("sentiment-analysis")

# In a production environment this is better (specifying a model):
# sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

print(sentiment_model("I love AI!"))
print(sentiment_model("I hate bugs."))
print(sentiment_model("I really don't like cakes"))
print(sentiment_model("But the cakes from my bride are delicious!"))
