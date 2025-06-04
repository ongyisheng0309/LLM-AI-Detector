import os
import fitz  # PyMuPDF

BASE_DIR = r"C:\Users\asus\Documents\Desmond's\Degree\HackAtk2.0\resume_dataset\data"  # Change this to your dataset root folder
OUTPUT_DIR = "spam_data/real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pdf_to_text(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"❌ Failed to read {pdf_path}: {e}")
        return ""

resume_counter = 1

# Walk through all subfolders
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(root, file)
            text = pdf_to_text(full_path)
            if text:
                out_file = os.path.join(OUTPUT_DIR, f"resume_{resume_counter}.txt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                resume_counter += 1

print(f"✅ Extracted {resume_counter - 1} resumes to {OUTPUT_DIR}/")
