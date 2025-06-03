from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from app.utils import pdf_to_text, highlight_pdf
from app.models import is_spam, compute_similarity, AIResumeDetector

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
UPLOAD_DIR = "resumes"
HIGHLIGHTED_DIR = "highlighted_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/relevance", response_class=HTMLResponse)
async def relevance_page(request: Request):
    return templates.TemplateResponse("relevance.html", {"request": request})

@app.get("/suspicious", response_class=HTMLResponse)
async def suspicious_page(request: Request):
    return templates.TemplateResponse("suspicious.html", {"request": request})

@app.post("/upload/")
async def upload_resumes(
    request: Request, 
    job_description: str = Form(...), 
    files: list[UploadFile] = File(...)
):
    resume_texts, names, ai_scores, highlighted_paths = [], [], [], []
    detector = AIResumeDetector()
    
    for file in files:
        try:
            file_path = f"{UPLOAD_DIR}/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            text = pdf_to_text(file_path)
            if not is_spam(text):
                resume_texts.append(text)
                names.append(file.filename)
                ai_result = detector.analyze_resume(text)
                ai_scores.append(ai_result['overall_score'])
                
                # Generate highlighted PDF
                highlighted_path = f"{HIGHLIGHTED_DIR}/highlighted_{file.filename}"
                highlight_pdf(file_path, highlighted_path, ai_result['highlighted_elements'])
                highlighted_paths.append(highlighted_path)
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue
    
    if resume_texts:
        results = compute_similarity(job_description, resume_texts)
        ranked = [(names[idx], round(score * 100, 1), round(ai_scores[idx] * 100, 1), highlighted_paths[idx]) for idx, score in results]
    else:
        ranked = []
    
    return templates.TemplateResponse("relevance.html", {"request": request, "results": ranked})

@app.post("/check-ai/")
async def check_ai_content(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        # Save uploaded file
        file_path = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Extract text and analyze
        text = pdf_to_text(file_path)
        detector = AIResumeDetector()
        ai_result = detector.analyze_resume(text)
        
        # Generate highlighted PDF
        highlighted_path = f"{HIGHLIGHTED_DIR}/highlighted_{file.filename}"
        highlight_pdf(file_path, highlighted_path, ai_result['highlighted_elements'])
        
        return templates.TemplateResponse("suspicious.html", {
            "request": request, 
            "ai_result": ai_result,
            "filename": file.filename
        })
        
    except Exception as e:
        print(f"Error processing {file.filename}: {e}")
        return templates.TemplateResponse("suspicious.html", {
            "request": request, 
            "error": f"Error processing file: {str(e)}"
        })

@app.get("/download/{filename:path}")
async def download_file(filename: str):
    file_path = f"{HIGHLIGHTED_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}