import pandas as pd
from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import json
import asyncio
from typing import Dict, Any
import uuid
from app.utils import pdf_to_text, highlight_pdf
from app.models import is_spam, compute_similarity, AIResumeDetector

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
UPLOAD_DIR = "resumes"
HIGHLIGHTED_DIR = "highlighted_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)

# Store progress for different tasks
progress_store: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/relevance", response_class=HTMLResponse)
async def relevance_page(request: Request):
    return templates.TemplateResponse("relevance.html", {"request": request})

@app.get("/suspicious", response_class=HTMLResponse)
async def suspicious_page(request: Request):
    return templates.TemplateResponse("suspicious.html", {"request": request})

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get progress for a specific task"""
    if task_id in progress_store:
        return progress_store[task_id]
    return {"progress": 0, "status": "not_found", "message": "Task not found"}

@app.post("/upload/")
async def upload_resumes(
    request: Request, 
    job_description: str = Form(...), 
    files: list[UploadFile] = File(...)
):
    # Generate unique task ID for this upload
    task_id = str(uuid.uuid4())
    progress_store[task_id] = {
        "progress": 0,
        "status": "starting",
        "message": "Initializing...",
        "current_file": "",
        "total_items": 0,
        "processed_items": 0
    }
    
    try:
        resume_texts, names, ai_scores, highlighted_paths = [], [], [], []
        detector = AIResumeDetector()
        total_items = 0
        processed_items = 0
        
        # First pass: count total items to process
        progress_store[task_id].update({
            "status": "counting",
            "message": "Counting files to process..."
        })
        
        for file in files:
            if file.filename.endswith('.csv'):
                # Read CSV to count rows
                file_content = await file.read()
                file.file.seek(0)  # Reset file pointer
                df = pd.read_csv(file.file)
                if 'Resume_str' in df.columns:
                    total_items += len(df)
            elif file.filename.endswith('.pdf'):
                total_items += 1
        
        progress_store[task_id].update({
            "total_items": total_items,
            "progress": 5,
            "status": "processing",
            "message": f"Processing {total_items} items..."
        })
        
        # Second pass: process files
        for file in files:
            try:
                progress_store[task_id]["current_file"] = file.filename
                
                if file.filename.endswith('.csv'):
                    file_content = await file.read()
                    file.file.seek(0)
                    df = pd.read_csv(file.file)
                    
                    if 'Resume_str' not in df.columns:
                        progress_store[task_id].update({
                            "status": "error",
                            "message": "CSV must contain 'Resume_str' column"
                        })
                        continue
                    
                    for idx, (_, row) in enumerate(df.iterrows()):
                        text = row['Resume_str']
                        if not is_spam(text):
                            resume_texts.append(text)
                            names.append(f"{file.filename} - {row['ID']}")
                            ai_result = detector.analyze_resume(text)
                            ai_scores.append(ai_result['overall_score'])
                            highlighted_paths.append(None)
                        
                        processed_items += 1
                        progress = 5 + (processed_items / total_items) * 85  # 5-90% for processing
                        progress_store[task_id].update({
                            "progress": int(progress),
                            "processed_items": processed_items,
                            "message": f"Processing resume {processed_items}/{total_items} from {file.filename}"
                        })
                        
                        # Allow other requests to be processed
                        await asyncio.sleep(0.01)
                
                elif file.filename.endswith('.pdf'):
                    file_path = f"{UPLOAD_DIR}/{file.filename}"
                    with open(file_path, "wb") as f:
                        f.write(await file.read())
                    
                    progress_store[task_id]["message"] = f"Extracting text from {file.filename}"
                    text = pdf_to_text(file_path)
                    
                    if not is_spam(text):
                        resume_texts.append(text)
                        names.append(file.filename)
                        
                        progress_store[task_id]["message"] = f"Analyzing AI content in {file.filename}"
                        ai_result = detector.analyze_resume(text)
                        ai_scores.append(ai_result['overall_score'])
                        
                        progress_store[task_id]["message"] = f"Creating highlighted PDF for {file.filename}"
                        highlighted_path = f"{HIGHLIGHTED_DIR}/highlighted_{file.filename}"
                        highlight_pdf(file_path, highlighted_path, ai_result['highlighted_elements'])
                        highlighted_paths.append(highlighted_path)
                    
                    processed_items += 1
                    progress = 5 + (processed_items / total_items) * 85
                    progress_store[task_id].update({
                        "progress": int(progress),
                        "processed_items": processed_items,
                        "message": f"Processed {processed_items}/{total_items} files"
                    })
                    
                    await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                processed_items += 1
                progress = 5 + (processed_items / total_items) * 85
                progress_store[task_id].update({
                    "progress": int(progress),
                    "processed_items": processed_items,
                    "message": f"Error processing {file.filename}, continuing..."
                })
                continue
        
        # Computing similarity scores
        progress_store[task_id].update({
            "progress": 90,
            "status": "finalizing",
            "message": "Computing similarity scores..."
        })
        
        if resume_texts:
            results = compute_similarity(job_description, resume_texts)
            ranked = [(names[idx], round(score * 100, 1), round(ai_scores[idx] * 100, 1), highlighted_paths[idx]) for idx, score in results]
        else:
            ranked = []
        
        progress_store[task_id].update({
            "progress": 100,
            "status": "completed",
            "message": "Processing completed successfully!"
        })
        
        # Clean up progress after a delay
        asyncio.create_task(cleanup_progress(task_id, 300))  # Clean up after 5 minutes
        
        return templates.TemplateResponse("relevance.html", {
            "request": request, 
            "results": ranked,
            "task_id": task_id
        })
    
    except Exception as e:
        progress_store[task_id].update({
            "progress": 0,
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        })
        print(f"General error: {e}")
        return templates.TemplateResponse("relevance.html", {
            "request": request, 
            "error": f"Error processing files: {str(e)}",
            "task_id": task_id
        })

@app.post("/check-ai/")
async def check_ai_content(
    request: Request,
    file: UploadFile = File(...)
):
    # Generate unique task ID for this upload
    task_id = str(uuid.uuid4())
    progress_store[task_id] = {
        "progress": 0,
        "status": "starting",
        "message": "Initializing...",
        "current_file": file.filename,
        "total_items": 1,
        "processed_items": 0
    }
    
    try:
        if file.filename.endswith('.csv'):
            progress_store[task_id].update({
                "progress": 10,
                "message": "Reading CSV file..."
            })
            
            df = pd.read_csv(file.file)
            if 'Resume_str' not in df.columns:
                progress_store[task_id].update({
                    "status": "error",
                    "message": "CSV must contain 'Resume_str' column"
                })
                return templates.TemplateResponse("suspicious.html", {
                    "request": request,
                    "error": "CSV must contain 'Resume_str' column",
                    "task_id": task_id
                })
            
            total_rows = len(df)
            progress_store[task_id].update({
                "total_items": total_rows,
                "progress": 15,
                "message": f"Processing {total_rows} resumes..."
            })
            
            results = []
            for idx, (_, row) in enumerate(df.iterrows()):
                text = row['Resume_str']
                detector = AIResumeDetector()
                ai_result = detector.analyze_resume(text)
                results.append({
                    'id': row['ID'],
                    'ai_result': ai_result,
                    'filename': f"{file.filename} - {row['ID']}"
                })
                
                progress = 15 + ((idx + 1) / total_rows) * 80  # 15-95%
                progress_store[task_id].update({
                    "progress": int(progress),
                    "processed_items": idx + 1,
                    "message": f"Analyzed {idx + 1}/{total_rows} resumes"
                })
                
                await asyncio.sleep(0.01)
            
            progress_store[task_id].update({
                "progress": 100,
                "status": "completed",
                "message": "Analysis completed!"
            })
            
            return templates.TemplateResponse("suspicious.html", {
                "request": request,
                "results": results,
                "task_id": task_id
            })
            
        elif file.filename.endswith('.pdf'):
            progress_store[task_id].update({
                "progress": 20,
                "message": "Saving PDF file..."
            })
            
            file_path = f"{UPLOAD_DIR}/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            progress_store[task_id].update({
                "progress": 40,
                "message": "Extracting text from PDF..."
            })
            
            text = pdf_to_text(file_path)
            
            progress_store[task_id].update({
                "progress": 60,
                "message": "Analyzing content for AI detection..."
            })
            
            detector = AIResumeDetector()
            ai_result = detector.analyze_resume(text)
            
            progress_store[task_id].update({
                "progress": 80,
                "message": "Creating highlighted PDF..."
            })
            
            highlighted_path = f"{HIGHLIGHTED_DIR}/highlighted_{file.filename}"
            highlight_pdf(file_path, highlighted_path, ai_result['highlighted_elements'])
            
            progress_store[task_id].update({
                "progress": 100,
                "status": "completed",
                "message": "Analysis completed!"
            })
            
            return templates.TemplateResponse("suspicious.html", {
                "request": request,
                "ai_result": ai_result,
                "filename": file.filename,
                "task_id": task_id
            })
        else:
            progress_store[task_id].update({
                "status": "error",
                "message": "Unsupported file type"
            })
            return templates.TemplateResponse("suspicious.html", {
                "request": request,
                "error": "Unsupported file type",
                "task_id": task_id
            })
            
    except Exception as e:
        progress_store[task_id].update({
            "progress": 0,
            "status": "error",
            "message": f"Error processing file: {str(e)}"
        })
        print(f"Error processing {file.filename}: {e}")
        return templates.TemplateResponse("suspicious.html", {
            "request": request,
            "error": f"Error processing file: {str(e)}",
            "task_id": task_id
        })

async def cleanup_progress(task_id: str, delay: int):
    """Clean up progress data after specified delay"""
    await asyncio.sleep(delay)
    if task_id in progress_store:
        del progress_store[task_id]

@app.get("/download/{filename:path}")
async def download_file(filename: str):
    file_path = f"{HIGHLIGHTED_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}