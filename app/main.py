from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from app.utils import pdf_to_text
from app.models import is_spam, compute_similarity

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
UPLOAD_DIR = "resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/upload/")
async def upload_resumes(request: Request, job_description: str = Form(...), files: list[UploadFile] = Form(...)):
    resume_texts, names = [], []
    for file in files:
        file_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        text = pdf_to_text(file_path)
        if not is_spam(text):
            resume_texts.append(text)
            names.append(file.filename)
    results = compute_similarity(job_description, resume_texts)
    ranked = [(names[idx], round(score, 3)) for idx, score in results]
    return templates.TemplateResponse("dashboard.html", {"request": request, "results": ranked})
