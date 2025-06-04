import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

model = SentenceTransformer('all-MiniLM-L6-v2')

class AIResumeDetector:
    def __init__(self):
        self.buzzwords = {"synergistic", "leveraged", "robust", "proactive", "strategic", 
                          "streamlined", "facilitated", "orchestrated", "spearheaded", 
                          "optimized", "utilized", "implemented", "enhanced"}
        self.vague_quantifiers = {"various", "several", "many", "numerous", "multiple", 
                                  "significant", "substantial", "a variety of"}
        self.passive_indicators = {"was", "were", "been", "by the", "by a", "by an"}

    def analyze_resume(self, text):
        results = {
            "overall_score": 0.0,
            "section_scores": {},
            "flagged_sections": [],
            "highlighted_elements": {"sentences": [], "words": []}
        }
        sections = self._split_into_sections(text)
        section_scores = []
        
        for section_name, content in sections.items():
            if not content.strip():
                continue
            section_results = {
                "name": section_name,
                "score": 0.0,
                "indicators": [],
                "evidence": []
            }
            self._check_generic_language(content, section_results, results["highlighted_elements"])
            self._check_structure_uniformity(content, section_results, results["highlighted_elements"])
            self._check_passive_voice(content, section_results, results["highlighted_elements"])
            self._check_vague_quantifiers(content, section_results, results["highlighted_elements"])
            self._check_specific_details(content, section_results, results["highlighted_elements"])
            
            if section_results["indicators"]:
                section_score = sum(ind["score"] for ind in section_results["indicators"]) 
                section_score /= len(section_results["indicators"])
                section_results["score"] = min(1.0, section_score)
                section_scores.append(section_results["score"])
            
            results["section_scores"][section_name] = section_results["score"]
            if section_results["score"] > 0.3:
                results["flagged_sections"].append(section_results)
        
        if section_scores:
            results["overall_score"] = min(1.0, sum(section_scores) / len(section_scores))
        
        return results

    def _split_into_sections(self, text):
        sections = {}
        current_section = "Header"
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^(experience|work history|employment):?$", line, re.I):
                current_section = "Experience"
            elif re.match(r"^(education|academic):?$", line, re.I):
                current_section = "Education"
            elif re.match(r"^(skills|technical skills):?$", line, re.I):
                current_section = "Skills"
            elif re.match(r"^(summary|objective|profile):?$", line, re.I):
                current_section = "Summary"
            else:
                sections.setdefault(current_section, "")
                sections[current_section] += line + "\n"
        return sections

    def _check_generic_language(self, content, results, highlighted_elements):
        words = re.findall(r'\b(\w+)\b', content.lower())
        buzzword_count = sum(1 for word in words if word in self.buzzwords)
        score = min(1.0, buzzword_count / 10)
        if score > 0.2:
            results["indicators"].append({"name": "generic_language", "score": score})
            results["evidence"].append(f"Found {buzzword_count} buzzwords")
            for word in words:
                if word in self.buzzwords:
                    highlighted_elements["words"].append(word)

    def _check_structure_uniformity(self, content, results, highlighted_elements):
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:
            return
        lengths = [len(line) for line in lines]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        score = 1.0 - min(1.0, variance / 100)
        if score > 0.7:
            results["indicators"].append({"name": "uniform_structure", "score": score})
            results["evidence"].append("Highly uniform line lengths")
            sentences = re.split(r'[.!?]+\s+', content)
            for sentence in sentences:
                if sentence.strip():
                    highlighted_elements["sentences"].append(sentence.strip())

    def _check_passive_voice(self, content, results, highlighted_elements):
        sentences = re.split(r'[.!?]+\s+', content)
        passive_count = 0
        for sentence in sentences:
            words = sentence.lower().split()
            if any(word in self.passive_indicators and i + 1 < len(words) and words[i + 1] in {"developed", "managed", "created"} for i, word in enumerate(words)):
                passive_count += 1
                highlighted_elements["sentences"].append(sentence.strip())
        score = min(1.0, passive_count / 5)
        if score > 0.3:
            results["indicators"].append({"name": "passive_voice", "score": score})
            results["evidence"].append(f"Found {passive_count} passive constructions")

    def _check_vague_quantifiers(self, content, results, highlighted_elements):
        words = re.findall(r'\b(\w+)\b', content.lower())
        vague_count = sum(1 for word in words if word in self.vague_quantifiers)
        score = min(1.0, vague_count / 5)
        if score > 0.2:
            results["indicators"].append({"name": "vague_quantifiers", "score": score})
            results["evidence"].append(f"Found {vague_count} vague quantifiers")
            for word in words:
                if word in self.vague_quantifiers:
                    highlighted_elements["words"].append(word)

    def _check_specific_details(self, content, results, highlighted_elements):
        has_dates = bool(re.search(r'\b(19|20)\d{2}\b', content))
        has_numbers = bool(re.search(r'\b\d+\b', content))
        score = 1.0 if not (has_dates or has_numbers) else 0.0
        if score > 0.5:
            results["indicators"].append({"name": "lack_details", "score": score})
            results["evidence"].append("Lacking specific dates or numbers")
            sentences = re.split(r'[.!?]+\s+', content)
            for sentence in sentences:
                if sentence.strip() and not (re.search(r'\b(19|20)\d{2}\b', sentence) or re.search(r'\b\d+\b', sentence)):
                    highlighted_elements["sentences"].append(sentence.strip())

def train_spam_model(spam_texts, real_texts):
    X = spam_texts + real_texts
    y = [1]*len(spam_texts) + [0]*len(real_texts)
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(X_vec, y)
    joblib.dump((vectorizer, clf), 'spam_model.joblib')

def is_spam(text):
    return False  # Placeholder

def compute_similarity(job_text, resume_texts):
    job_vec = model.encode(job_text, convert_to_tensor=True)
    results = []
    for idx, text in enumerate(resume_texts):
        resume_vec = model.encode(text, convert_to_tensor=True)
        sim = util.cos_sim(job_vec, resume_vec).item()
        results.append((idx, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)