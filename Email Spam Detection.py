# Email Spam Detection
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (Replace with real CSV: columns ['label', 'message'])
data = pd.DataFrame({
    'label': ['ham', 'spam', 'ham', 'spam'],
    'message': [
        'Hey, are we meeting today?',
        'Congratulations! You have won a free ticket',
        'I will call you later',
        'Free entry in 2 a weekly competition!'
    ]
})

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
