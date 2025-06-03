# LLM-AI-Detector: AI-Powered Talent Acquisition System

This is a prototype web application built with **FastAPI** that streamlines the recruitment process using AI. It parses PDF resumes, filters out spam, and ranks candidates by matching them with a given job description using **MiniLM embeddings**.

---

## 🚀 Features

- Upload multiple PDF resumes
- Input a job description and rank candidate matches
- Automatically filters out spam/fake resumes
- Uses SentenceTransformer (MiniLM-L6-v2) for semantic similarity
- Lightweight FastAPI frontend with auto-refresh on code changes

---

## 🛠 Tech Stack

- **Python 3.10**
- **FastAPI** (backend API + simple UI)
- **PyMuPDF** (PDF to text)
- **sentence-transformers** (`all-MiniLM-L6-v2`)
- **scikit-learn** (spam detection)
- **spaCy** (optional NLP pipeline)

---

## 📦 Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/ongyisheng0309/LLM-AI-Detector.git

### 3. Install Dependencies

pip install -r requirements.txt


### 4. Run the App

uvicorn app.main:app --reload
Then open your browser to:
http://127.0.0.1:8000/

🧪 Usage Instructions
Paste a job description into the text area.

Upload multiple PDF resumes.

Click "Upload and Match".

View the ranked candidates based on similarity.
