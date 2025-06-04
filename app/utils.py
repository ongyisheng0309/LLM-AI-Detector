import fitz  # PyMuPDF
import re

def pdf_to_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def highlight_pdf(input_path, output_path, highlighted_elements):
    doc = fitz.open(input_path)
    for page in doc:
        # Highlight sentences
        for sentence in highlighted_elements["sentences"]:
            text_instances = page.search_for(sentence)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=[1, 1, 0])  # Yellow for sentences
                highlight.update()
        # Highlight words
        for word in highlighted_elements["words"]:
            text_instances = page.search_for(word)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=[1, 0, 0])  # Red for words
                highlight.update()
    doc.save(output_path)
    doc.close()