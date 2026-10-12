# Universal AI-Powered Resume Screening & Ranking System

## 1. Project Overview
This project is an AI-powered Resume Screening and Candidate Ranking tool built with Python, NLTK, Scikit-Learn, and Streamlit. It accepts **any real-world candidate resume** in **PDF, TXT, or DOCX** format alongside **any Job Description (JD)**, automatically extracts **Skills, Education, and Experience**, computes a relevance score, and generates a ranked candidate leaderboard and downloadable CSV report.

---

## 2. Key Features
* **Multi-Format Resume Support**: Parses single-page and multi-page resumes in `.pdf` (via `pdfplumber` and `PyPDF2`), `.txt`, and `.docx` formats.
* **Universal Section & Entity Extraction**:
  * **Candidate Name**: Automatically identifies candidate names from unstructured resume headers or explicit labels.
  * **Dynamic + Taxonomy Skill Extraction**: Combines a 150+ multi-domain skill database with dynamic keyword extraction from the uploaded Job Description and the resume's `Skills` section.
  * **Education Extraction**: Detects degrees (`B.Tech`, `B.E.`, `M.Tech`, `B.Sc`, `M.Sc`, `BCA`, `MCA`, `MBA`, `Ph.D.`, etc.) and academic qualifications.
  * **Experience Calculation**: Extracts total years of experience from both explicit mentions (e.g., `"2 years of experience"`) and employment date ranges (e.g., `"May 2024 - Aug 2024"`, `"Jan 2023 - Present"`).
* **Dual Interface (Streamlit Web UI + CLI)**:
  * **Streamlit Web App (`app.py`)**: Drag-and-drop multi-file resume uploader, custom Job Description input, interactive leaderboard table, expandable candidate summary cards, and one-click CSV report download.
  * **Command-Line Interface (`resume_screener.py`)**: Supports batch screening of any file or folder from the terminal and auto-generates sample `.pdf` and `.txt` resumes for instant testing.

---

## 3. Project Structure
```text
resume-screening-ai/
├── app.py                         # Streamlit Web UI for uploading any resume & JD
├── resume_screener.py             # Core NLP/TF-IDF engine, CLI runner & sample generator
├── candidate_ranking_report.csv   # Generated CSV ranking report
├── README.md                      # Project documentation & scoring algorithm explanation
└── sample_data/                   # Auto-generated sample resumes (.pdf & .txt) and JD
    ├── job_description.txt
    ├── resume_1_arjun_mehta.pdf
    ├── resume_2_priya_sharma.pdf
    └── resume_3_rohan_verma.txt