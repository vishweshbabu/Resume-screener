import os
import re
import csv
import zipfile
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)


# Curated multi-domain technical & professional skill vocabulary
BASE_SKILL_KEYWORDS = {
    # AI / ML / GenAI / Data Science / CV
    "artificial intelligence", "ai", "machine learning", "ml",
    "deep learning", "nlp", "natural language processing",
    "computer vision", "generative ai", "genai", "llm", "llms",
    "large language models", "langchain", "rag",
    "retrieval augmented generation", "prompt engineering",
    "object detection", "image processing", "image masking",
    "voice processing", "document processing", "knowledge retrieval",
    "yolo", "yolov5", "yolov8", "comfyui", "groundingdino", "flux",
    "stable diffusion", "opencv", "tensorflow", "pytorch", "keras",
    "scikit-learn", "sklearn", "huggingface", "transformers",
    "nltk", "spacy", "pandas", "numpy", "scipy", "matplotlib",
    "seaborn", "plotly", "data analysis", "data science",
    "data visualization", "feature engineering", "statistics",
    # Programming & Web & Frameworks
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "nosql",
    "html", "css", "react", "angular", "vue", "node.js",
    "django", "flask", "fastapi", "streamlit", "rest api", "json",
    # Cloud, DevOps & Tools
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "linux", "ci/cd", "jenkins", "google colab", "jupyter",
    # Core Engineering / Electronics / IoT
    "embedded systems", "iot", "internet of things", "arduino",
    "raspberry pi", "microcontrollers", "plc", "scada", "vlsi",
    "verilog", "vhdl", "pcb design", "matlab", "simulink", "autocad",
    "signal processing", "telecommunications", "robotics",
    # Business, Product & Soft Skills
    "product management", "project management", "business analysis",
    "agile", "scrum", "jira", "tableau", "power bi", "excel",
    "sap", "erp", "crm", "salesforce", "digital marketing", "seo",
    "customer support", "communication", "leadership",
    "problem solving", "teamwork", "debugging", "testing"
}

# Words that should NEVER be mistaken for a candidate's name
NON_NAME_WORDS = {
    "professional", "summary", "executive", "profile", "objective",
    "career", "resume", "curriculum", "vitae", "cv", "biodata",
    "experience", "work", "employment", "history", "internship",
    "internships", "education", "academic", "qualification",
    "qualifications", "skills", "technical", "competencies",
    "projects", "personal", "key", "certifications", "achievements",
    "awards", "contact", "details", "information", "declaration",
    "references", "languages", "interests", "hobbies", "about",
    "engineer", "developer", "analyst", "associate", "manager",
    "specialist", "consultant", "intern", "fresher", "candidate",
    "artificial", "intelligence", "machine", "learning", "data",
    "science", "software", "full", "stack", "frontend", "backend"
}

SECTION_HEADERS = {
    "summary": (
        r"^\s*(?:professional\s+summary|executive\s+summary|"
        r"summary|profile|career\s+objective|objective|about\s+me)"
        r"\s*[:\-]?\s*$"
    ),
    "education": (
        r"^\s*(?:education|academic\s+background|academics|"
        r"qualifications|educational\s+qualifications?)\s*[:\-]?\s*$"
    ),
    "experience": (
        r"^\s*(?:work\s+experience|professional\s+experience|"
        r"experience|employment\s+history|internships?|"
        r"industrial\s+training|work\s+history)\s*[:\-]?\s*$"
    ),
    "projects": (
        r"^\s*(?:projects?|academic\s+projects?|"
        r"personal\s+projects?|key\s+projects?)\s*[:\-]?\s*$"
    ),
    "skills": (
        r"^\s*(?:technical\s+skills|skills|core\s+competencies|"
        r"key\s+skills|expertise|technologies|"
        r"tools\s*(?:&|and)\s*technologies)\s*[:\-]?\s*$"
    ),
    "other": (
        r"^\s*(?:certifications?|achievements?|awards?|"
        r"publications?|languages?|hobbies|interests|"
        r"references|declaration)\s*[:\-]?\s*$"
    ),
}

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12
}


class ResumeScreenerAI:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def extract_text_from_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        if ext == ".txt":
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        return f.read()
                except Exception:
                    continue

        elif ext == ".pdf":
            if pdfplumber is not None:
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                except Exception:
                    pass

            if not text.strip() and PyPDF2 is not None:
                try:
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                except Exception:
                    pass

        elif ext == ".docx":
            try:
                with zipfile.ZipFile(file_path) as zf:
                    xml_content = zf.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                paragraphs = []
                for elem in tree.iter():
                    if elem.tag.endswith("}p"):
                        texts = [
                            t.text for t in elem.iter()
                            if t.tag.endswith("}t") and t.text
                        ]
                        if texts:
                            paragraphs.append("".join(texts))
                text = "\n".join(paragraphs)
            except Exception:
                pass

        return text

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
        tokens = nltk.word_tokenize(text)
        cleaned = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 1
        ]
        return " ".join(cleaned)

    def split_sections(self, raw_text):
        sections = {
            "header": [], "summary": [], "education": [],
            "experience": [], "projects": [], "skills": [], "other": []
        }
        current_sec = "header"

        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            cleaned_hdr = re.sub(r"^[•\-*#\d.)\s]+", "", stripped).strip()
            matched_sec = None
            if len(cleaned_hdr) <= 42:
                for sec_name, pattern in SECTION_HEADERS.items():
                    if re.match(pattern, cleaned_hdr, flags=re.IGNORECASE):
                        matched_sec = sec_name
                        break

            if matched_sec:
                current_sec = matched_sec
            else:
                sections[current_sec].append(stripped)

        return {k: "\n".join(v) for k, v in sections.items()}

    # -----------------------------------------------------------------
    # FIXED: Candidate Name Extraction (Never picks "Professional Summary")
    # -----------------------------------------------------------------
    def extract_candidate_name(self, raw_text, filename):
        explicit = re.search(
            r"^\s*Name\s*[:\-]\s*([A-Za-z][A-Za-z\s.]{1,40})",
            raw_text,
            re.IGNORECASE | re.MULTILINE
        )
        if explicit:
            return explicit.group(1).strip()

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

        for line in lines[:10]:
            # Take first segment if separated by | or - or ,
            first_part = re.split(r"[|•–—\-_]", line)[0].strip()
            clean_line = re.sub(r"[^A-Za-z.\s]", " ", first_part)
            clean_line = re.sub(r"\s+", " ", clean_line).strip()

            low_line = clean_line.lower()
            if any(
                x in line.lower()
                for x in ["@", "http", "www.", "linkedin", "github", "+91"]
            ):
                continue

            words = clean_line.split()
            if not (1 <= len(words) <= 5):
                continue

            # Reject if ANY word is a section title or role keyword
            if any(w.lower().strip(".") in NON_NAME_WORDS for w in words):
                continue

            if all(re.match(r"^[A-Za-z.]+$", w) for w in words):
                return " ".join(
                    w.upper() if len(w.strip(".")) == 1 else w.capitalize()
                    for w in words
                )

        # Fallback: Extract person's name cleanly from filename
        base = os.path.splitext(os.path.basename(filename))[0]
        tokens = re.split(r"[_\-\s]+", base)
        name_tokens = []
        for tok in tokens:
            if tok.lower() in NON_NAME_WORDS or tok.isdigit():
                break
            if tok.isalpha():
                name_tokens.append(tok.capitalize())
        if name_tokens:
            return " ".join(name_tokens)

        return "Candidate"

    # -----------------------------------------------------------------
    # FIXED: Clean Skill Extraction (No random sentence fragments)
    # -----------------------------------------------------------------
    def build_dynamic_skill_set(self, jd_text):
        dynamic_skills = set(BASE_SKILL_KEYWORDS)
        # Add clean technical tokens from the JD
        jd_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9+#.\-]{1,22}\b", jd_text)
        for tok in jd_tokens:
            low = tok.lower().strip(".-")
            if low in BASE_SKILL_KEYWORDS:
                dynamic_skills.add(low)
            elif (
                any(c.isupper() for c in tok[1:])
                or any(c in tok for c in "+#.")
            ) and low not in self.stop_words:
                dynamic_skills.add(low)
        return dynamic_skills

    def extract_skills(self, raw_text, custom_skill_pool=None):
        skill_pool = (
            custom_skill_pool
            if custom_skill_pool is not None
            else BASE_SKILL_KEYWORDS
        )
        text_lower = raw_text.lower()
        found_skills = set()

        for skill in skill_pool:
            pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
            if re.search(pattern, text_lower):
                found_skills.add(skill)

        # Normalize duplicates (e.g. keep 'llm' if both 'llm' and 'llms' found)
        if "llm" in found_skills and "llms" in found_skills:
            found_skills.remove("llms")

        return sorted(found_skills)

    # -----------------------------------------------------------------
    # FIXED: Strict Degree & Education Extraction (Never matches "before")
    # -----------------------------------------------------------------
    def extract_education(self, raw_text):
        sections = self.split_sections(raw_text)
        edu_block = sections.get("education", "").strip()
        search_target = edu_block if edu_block else raw_text

        # Case-safe degree patterns that will NEVER match "before" or "December"
        strict_degree_patterns = [
            (
                r"\b(?:B\.\s*Tech|B\.Tech|M\.\s*Tech|M\.Tech|"
                r"B\.\s*E\.|B\.E\.|M\.\s*E\.|M\.E\.|"
                r"B\.\s*Sc|B\.Sc|M\.\s*Sc|M\.Sc|"
                r"B\.\s*Com|M\.\s*Com|BCA|MCA|BBA|MBA|Ph\.?\s*D)"
                r"[^\n;|•]*"
            ),
            (
                r"\b(?:Bachelor\s+of\s+[A-Za-z\s&]+|"
                r"Master\s+of\s+[A-Za-z\s&]+|"
                r"Diploma\s+in\s+[A-Za-z\s&]+)[^\n;|•]*"
            ),
        ]

        found_degrees = []
        for pat in strict_degree_patterns:
            for match in re.findall(pat, search_target, flags=re.IGNORECASE):
                cleaned = re.sub(r"\s+", " ", match).strip(" ,;-•|")
                # Ensure it didn't match an English word like "be" or "me"
                first_word = cleaned.split()[0].lower()
                if first_word in {"be", "me", "ma", "ms", "ba", "bs", "before"}:
                    continue
                if 4 <= len(cleaned) <= 100 and cleaned not in found_degrees:
                    found_degrees.append(cleaned)

        if found_degrees:
            return "; ".join(found_degrees[:2])

        # Fallback: take clean lines from the Education section
        if edu_block:
            edu_lines = [
                line.strip(" •-*")
                for line in edu_block.splitlines()
                if len(line.strip()) > 5
            ]
            if edu_lines:
                return " | ".join(edu_lines[:2])

        return "B.Tech / Engineering Graduate (See Resume)"

    def extract_experience(self, raw_text):
        total_years = 0.0
        yr_pat = (
            r"(\d+(?:\.\d+)?)\+?\s*years?"
            r"(?:\s+of)?\s+(?:experience|exp|work|industry)"
        )
        mo_pat = (
            r"(\d+)\s*months?\s+(?:of\s+)?"
            r"(?:experience|internship|training)"
        )
        yr_matches = re.findall(yr_pat, raw_text, re.IGNORECASE)
        mo_matches = re.findall(mo_pat, raw_text, re.IGNORECASE)

        if yr_matches:
            valid_yrs = [float(y) for y in yr_matches if float(y) < 40]
            if valid_yrs:
                total_years = max(valid_yrs)

        sections = self.split_sections(raw_text)

        if total_years == 0.0:
            now_yr, now_mo = datetime.now().year, datetime.now().month
            month_names = (
                r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
                r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
                r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
            )
            date_range_pat = (
                rf"({month_names})\s+(20\d{{2}}|19\d{{2}})"
                rf"\s*(?:[-–—]|to)\s*"
                rf"(?:({month_names})\s+)?"
                rf"(20\d{{2}}|19\d{{2}}|Present|Current|Now)"
            )
            months_accum = 0
            exp_block = sections.get("experience", "") or raw_text
            for m1, y1, m2, y2 in re.findall(
                date_range_pat, exp_block, flags=re.IGNORECASE
            ):
                start_yr = int(y1)
                start_mo = MONTH_MAP.get(m1.lower()[:3], 1) if m1 else 1
                if y2.lower() in ("present", "current", "now"):
                    end_yr, end_mo = now_yr, now_mo
                else:
                    end_yr = int(y2)
                    end_mo = MONTH_MAP.get(m2.lower()[:3], 12) if m2 else 12
                diff = (end_yr - start_yr) * 12 + (end_mo - start_mo) + 1
                if 1 <= diff <= 120:
                    months_accum += diff
            if months_accum > 0:
                total_years = round(months_accum / 12.0, 1)

        if total_years == 0.0 and mo_matches:
            valid_mos = [float(m) for m in mo_matches if float(m) <= 60]
            if valid_mos:
                total_years = round(sum(valid_mos) / 12.0, 1)

        exp_lines = [
            l.strip(" •-*")
            for l in sections.get("experience", "").splitlines()
            if len(l.strip()) > 6
        ]
        proj_lines = [
            l.strip(" •-*")
            for l in sections.get("projects", "").splitlines()
            if len(l.strip()) > 6
        ]

        if exp_lines:
            summary = " | ".join(exp_lines[:2])
        elif proj_lines:
            summary = "Projects: " + " | ".join(proj_lines[:2])
        elif total_years > 0:
            summary = f"{total_years} years of professional experience"
        else:
            summary = "Fresher / Entry-Level Candidate"

        return total_years, summary[:160]

    def screen_and_rank(self, jd_text, resume_files):
        dynamic_pool = self.build_dynamic_skill_set(jd_text)
        jd_skills = set(
            self.extract_skills(jd_text, custom_skill_pool=dynamic_pool)
        )
        preprocessed_jd = self.preprocess_text(jd_text)

        candidates = []
        preprocessed_resumes = []

        for file_path in resume_files:
            raw_text = self.extract_text_from_file(file_path)
            if not raw_text.strip():
                continue

            prep_resume = self.preprocess_text(raw_text)
            preprocessed_resumes.append(prep_resume)

            name = self.extract_candidate_name(raw_text, file_path)
            resume_skills = self.extract_skills(
                raw_text, custom_skill_pool=dynamic_pool
            )
            education = self.extract_education(raw_text)
            exp_years, exp_summary = self.extract_experience(raw_text)

            matched_skills = sorted(
                set(resume_skills).intersection(jd_skills)
            )
            skill_match_pct = (
                (len(matched_skills) / len(jd_skills) * 100.0)
                if jd_skills else 0.0
            )

            candidates.append({
                "file": os.path.basename(file_path),
                "name": name,
                "skills": resume_skills,
                "matched_skills": matched_skills,
                "skill_match_score": round(skill_match_pct, 2),
                "education": education,
                "experience_years": exp_years,
                "experience_summary": exp_summary,
            })

        if not candidates:
            return [], sorted(jd_skills)

        corpus = [preprocessed_jd] + preprocessed_resumes
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)

        cosine_scores = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        ).flatten()

        for idx, cand in enumerate(candidates):
            tfidf_pct = round(float(cosine_scores[idx]) * 100.0, 2)
            cand["tfidf_similarity"] = tfidf_pct
            if jd_skills:
                overall = (
                    (0.50 * tfidf_pct) + (0.50 * cand["skill_match_score"])
                )
            else:
                overall = tfidf_pct
            cand["overall_match_score"] = round(overall, 2)

        candidates.sort(
            key=lambda x: x["overall_match_score"], reverse=True
        )
        for rank_idx, cand in enumerate(candidates, 1):
            cand["rank"] = rank_idx

        return candidates, sorted(jd_skills)


def create_minimal_pdf(file_path, text_content):
    lines = text_content.strip().split("\n")
    stream_lines = ["BT", "/F1 10 Tf", "50 760 Td", "14 TL"]
    for line in lines:
        safe = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        stream_lines.append(f"({safe}) Tj T*")
    stream_lines.append("ET")
    stream_data = "\n".join(stream_lines).encode("latin-1", errors="ignore")

    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n"
    )
    obj4 = (
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >>\nendobj\n"
    )
    obj5 = (
        f"5 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
        + stream_data
        + b"\nendstream\nendobj\n"
    )

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        obj3,
        obj4,
        obj5,
    ]
    pdf_bytes = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf_bytes))
        pdf_bytes += obj
    xref_offset = len(pdf_bytes)
    xref_hdr = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    pdf_bytes += xref_hdr.encode("latin-1")
    for off in offsets:
        pdf_bytes += f"{off:010d} 00000 n \n".encode("latin-1")

    trailer_str = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    pdf_bytes += trailer_str.encode("latin-1")
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)


def generate_sample_dataset(data_dir="sample_data"):
    os.makedirs(data_dir, exist_ok=True)
    jd_lines = [
        "Job Title: AI / Machine Learning Engineer",
        "We are looking for an AI / Machine Learning Engineer "
        "to build NLP, Computer Vision, and Generative AI applications.",
        "Required Skills: Python, Machine Learning, Deep Learning, "
        "NLP, Computer Vision, Generative AI, LLM, RAG, OpenCV, YOLO, "
        "TensorFlow, PyTorch, Scikit-learn, Streamlit, Git.",
        "Education: B.Tech / B.E. in Computer Science, AI, "
        "Electronics, or related field."
    ]
    jd_path = os.path.join(data_dir, "job_description.txt")
    with open(jd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jd_lines))

    r1 = "\n".join([
        "ARJUN MEHTA",
        "arjun.mehta@email.com | +91-9876543210",
        "EDUCATION",
        "B.Tech in Computer Science and Engineering, 2023",
        "EXPERIENCE",
        "Machine Learning Engineer at TechNova AI (Jan 2023 - Present)",
        "Built LLM chatbots using LangChain, RAG, HuggingFace, PyTorch.",
        "SKILLS",
        "Python, SQL, Machine Learning, Deep Learning, NLP, Generative AI, "
        "LLM, LangChain, RAG, PyTorch, TensorFlow, Scikit-learn, FastAPI, "
        "Streamlit, Docker, Git"
    ])
    r2 = "\n".join([
        "PRIYA SHARMA",
        "priya.sharma@email.com",
        "EDUCATION",
        "M.Tech in Data Science, 2024",
        "EXPERIENCE",
        "Data Scientist at FinSight Analytics (2 years of experience)",
        "SKILLS",
        "Python, SQL, Machine Learning, NLP, TensorFlow, Scikit-learn, "
        "Pandas, NumPy, Flask, Git, Data Analysis"
    ])
    r3 = "\n".join([
        "Rohan Verma",
        "rohan.verma@email.com",
        "EDUCATION",
        "B.Tech in Information Technology, 2025",
        "EXPERIENCE",
        "AI Intern at VisionLabs (May 2025 - Aug 2025)",
        "SKILLS",
        "Python, C++, Machine Learning, Deep Learning, Computer Vision, "
        "PyTorch, Keras, Flask, Streamlit, Git"
    ])

    samples = [
        ("resume_1_arjun_mehta.pdf", r1),
        ("resume_2_priya_sharma.pdf", r2),
        ("resume_3_rohan_verma.txt", r3),
    ]
    for fname, content in samples:
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".pdf"):
            create_minimal_pdf(fpath, content)
        else:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
    return jd_path, data_dir


def export_to_csv(candidates, output_csv="candidate_ranking_report.csv"):
    fieldnames = [
        "Rank", "Candidate Name", "Resume File",
        "Overall Match Score (%)", "TF-IDF Similarity (%)",
        "Skill Match (%)", "Experience (Years)",
        "Education", "Matched JD Skills", "All Extracted Skills"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "Rank": c["rank"],
                "Candidate Name": c["name"],
                "Resume File": c["file"],
                "Overall Match Score (%)": f"{c['overall_match_score']:.2f}",
                "TF-IDF Similarity (%)": f"{c['tfidf_similarity']:.2f}",
                "Skill Match (%)": f"{c['skill_match_score']:.2f}",
                "Experience (Years)": c["experience_years"],
                "Education": c["education"],
                "Matched JD Skills": ", ".join(c["matched_skills"]),
                "All Extracted Skills": ", ".join(c["skills"]),
            })


def collect_resume_files(path_input):
    valid_exts = (".pdf", ".txt", ".docx")
    if os.path.isfile(path_input) and path_input.lower().endswith(valid_exts):
        return [path_input]
    elif os.path.isdir(path_input):
        return [
            os.path.join(path_input, f)
            for f in sorted(os.listdir(path_input))
            if f.lower().endswith(valid_exts)
            and f.lower() != "job_description.txt"
        ]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Universal AI Resume Screener"
    )
    parser.add_argument("--jd", type=str, help="Path to JD file or text")
    parser.add_argument("--resumes", nargs="+", help="Resume files/folders")
    parser.add_argument(
        "--output",
        type=str,
        default="candidate_ranking_report.csv",
        help="Output CSV filename"
    )
    args = parser.parse_args()

    resume_files = []
    if args.resumes:
        for item in args.resumes:
            resume_files.extend(collect_resume_files(item))

    if args.jd:
        if os.path.isfile(args.jd):
            with open(args.jd, "r", encoding="utf-8", errors="ignore") as f:
                jd_text = f.read()
        else:
            jd_text = args.jd
    else:
        jd_text = ""

    if not resume_files or not jd_text.strip():
        jd_path, sample_dir = generate_sample_dataset("sample_data")
        if not jd_text.strip():
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
        if not resume_files:
            resume_files = collect_resume_files(sample_dir)

    screener = ResumeScreenerAI()
    candidates, _ = screener.screen_and_rank(jd_text, resume_files)

    print("\n" + "=" * 90)
    print("                    AI RESUME SCREENING - CANDIDATE RANKING TABLE")
    print("=" * 90)
    hdr = (
        f"{'Rank':<5} | {'Candidate Name':<20} | {'Type':<5} | "
        f"{'Overall':<10} | {'TF-IDF':<10} | {'Skills':<10} | {'Exp'}"
    )
    print(hdr)
    print("-" * 90)
    for c in candidates:
        ext = os.path.splitext(c["file"])[1].upper().replace(".", "")
        row = (
            f"#{c['rank']:<4} | {c['name'][:20]:<20} | {ext:<5} | "
            f"{c['overall_match_score']:>8.2f}% | "
            f"{c['tfidf_similarity']:>8.2f}% | "
            f"{c['skill_match_score']:>8.2f}% | "
            f"{c['experience_years']} yrs"
        )
        print(row)
    print("=" * 90)

    export_to_csv(candidates, args.output)
    print(f"\n[SUCCESS] Saved ranking report to '{args.output}'")


if __name__ == "__main__":
    main()