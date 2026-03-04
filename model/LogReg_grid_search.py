from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import csv

# Replace the hardcoded data list with this:
with open('train.csv', encoding='utf-8') as f:
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
    ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
])

param_grid = {
    'tfidf__ngram_range': [(2, 4), (2, 5), (3, 5)],
    'tfidf__max_features': [50_000, 100_000],
    'tfidf__min_df': [1, 2],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.1, 0.5, 1.0, 5.0],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',   # good for imbalanced classes
    n_jobs=-1,            # use all CPU cores
    verbose=2
)

grid_search.fit(X_train, y_train)

print('Best params:', grid_search.best_params_)
print('Best CV f1_macro:', grid_search.best_score_.round(3))

# Evaluate best model on test set
y_pred = grid_search.best_estimator_.predict(X_test)
print('\n=== Classification Report (best model) ===')
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Save best model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

print('\nBest model saved to sentiment_model.pkl')