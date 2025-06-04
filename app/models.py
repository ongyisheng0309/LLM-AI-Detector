from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def train_spam_model(spam_texts, real_texts):
    X = spam_texts + real_texts
    y = [1]*len(spam_texts) + [0]*len(real_texts)
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(X_vec, y)
    joblib.dump((vectorizer, clf), 'spam_model.joblib')

def is_spam(text):
    vectorizer, clf = joblib.load('spam_model.joblib')
    return clf.predict(vectorizer.transform([text]))[0] == 1

def compute_similarity(job_text, resume_texts):
    job_vec = model.encode(job_text, convert_to_tensor=True)
    results = []
    for idx, text in enumerate(resume_texts):
        resume_vec = model.encode(text, convert_to_tensor=True)
        sim = util.cos_sim(job_vec, resume_vec).item()
        results.append((idx, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)
