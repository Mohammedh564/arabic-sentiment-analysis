# baseline_lr.py
"""
Baseline Logistic Regression for Arabic Sentiment Analysis
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# --------------------------
# Paths
# --------------------------
DATA_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE   = os.path.join(DATA_DIR, "val.csv")
TEST_FILE  = os.path.join(DATA_DIR, "test.csv")  # optional

# --------------------------
# Load data
# --------------------------
train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)

X_train = train_df['text'].values
y_train = train_df['label'].values
X_val   = val_df['text'].values
y_val   = val_df['label'].values

# --------------------------
# TF-IDF Vectorization
# --------------------------
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    min_df=5
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)

# --------------------------
# Train Logistic Regression
# --------------------------
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight='balanced'  # helps with class imbalance
)
model.fit(X_train_tfidf, y_train)

# --------------------------
# Evaluate
# --------------------------
y_pred = model.predict(X_val_tfidf)
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# --------------------------
# Save Model & Vectorizer
# --------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_lr.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved in ./models/")