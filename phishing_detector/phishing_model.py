# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 2. LOAD DATASET
# =========================

df = pd.read_csv("emails.csv")

X = df["text"]
y = df["label"]

print("\nDataset Loaded Successfully")
print(df.head())


# =========================
# 3. FEATURE ENGINEERING (URL FEATURES)
# =========================

class URLFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        features = []

        suspicious_words = [
            "verify", "login", "password", "bank",
            "urgent", "click", "update", "free",
            "winner", "secure", "account"
        ]

        for email in X:

            email = str(email)

            # Extract URLs
            urls = re.findall(r'https?://\S+|www\.\S+', email)
            url_count = len(urls)

            # Suspicious keyword count
            keyword_count = sum(word in email.lower() for word in suspicious_words)

            # Long URL detection
            long_urls = sum(len(url) > 30 for url in urls)

            features.append([url_count, keyword_count, long_urls])

        return np.array(features)


# =========================
# 4. FEATURE COMBINATION
# =========================

features = FeatureUnion([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),

    ("url_features", Pipeline([
        ("extract", URLFeatureExtractor()),
        ("scale", StandardScaler())
    ]))
])


# =========================
# 5. MODEL PIPELINE
# =========================

model = Pipeline([
    ("features", features),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


# =========================
# 6. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =========================
# 7. TRAIN MODEL
# =========================

model.fit(X_train, y_train)


# =========================
# 8. PREDICTIONS
# =========================

y_pred = model.predict(X_test)


# =========================
# 9. EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================")
print("MODEL EVALUATION")
print("====================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# 10. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Safe", "Phishing"],
            yticklabels=["Safe", "Phishing"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =========================
# 11. TEST NEW EMAIL
# =========================

new_email = """
URGENT!
Your account has been suspended.
Click https://secure-login-update.com to verify immediately.
"""

prediction = model.predict([new_email])[0]

print("\n====================")
print("NEW EMAIL PREDICTION")
print("====================")
print("Result:", prediction.upper())