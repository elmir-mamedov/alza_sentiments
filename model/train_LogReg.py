#train_LogReg.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import csv

# Replace the hardcoded data list with this:
with open('../get_data/reviews.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data = [{'text': row['text'], 'label': int(row['label'])} for row in reader]

texts = [d['text'] for d in data]
labels = [d['label'] for d in data]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb',   # character n-grams — great for morphologically rich langs
        ngram_range=(2, 4),
        min_df=1,
        max_features=10000
    )),
    ('clf', LogisticRegression(max_iter=1000, C=1.0))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Save model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\nModel saved to sentiment_model.pkl")