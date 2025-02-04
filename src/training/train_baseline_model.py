import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ..preprocessing.text_cleaning import preprocess_text

try:
    df = pd.read_csv("data/raw/imdb_dataset.csv")
except FileNotFoundError:
    raise FileNotFoundError("Dataset not found! Ensure 'imdb_dataset.csv' exists in 'data/raw/'.")

df['cleaned_review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words=None)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Top TF-IDF Features:", vectorizer.get_feature_names_out()[:50])

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

joblib.dump(model, 'models/baseline_model/baseline_model.pkl')
joblib.dump(vectorizer, 'models/baseline_model/tfidf_vectorizer.pkl')

print("Baseline model trained and saved successfully!")