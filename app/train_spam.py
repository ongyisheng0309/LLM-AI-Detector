import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def load_texts(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

real = load_texts("spam_data/real")
spam = load_texts("spam_data/spam")

X = spam + real
y = [1]*len(spam) + [0]*len(real)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

clf = LogisticRegression()
clf.fit(X_vec, y)

joblib.dump((vectorizer, clf), "spam_model.joblib")
print("✅ Trained spam_model.joblib saved")
