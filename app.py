import os
import tempfile
import pandas as pd
import streamlit as st
from resume_screener import ResumeScreenerAI, generate_sample_dataset

st.set_page_config(page_title="AI Resume Screening Tool", page_icon="📄", layout="wide")

st.title("📄 Universal AI Resume Screening & Ranking System")
st.markdown(
    "Upload **any candidate resume(s)** in **PDF, TXT, or DOCX** format and enter **any Job Description** "
    "to automatically extract skills, education, and experience, and rank candidates."
)

screener = ResumeScreenerAI()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Enter Any Job Description")
    jd_text = st.text_area(
        "Paste the target Job Description (or upload a JD text file below):",
        height=240,
        placeholder="Paste any Job Description here (e.g., AI Engineer, Full Stack Developer, Data Analyst, Marketing Manager)..."
    )
    jd_file = st.file_uploader("Or upload a Job Description file (.txt or .pdf)", type=["txt", "pdf", "docx"], key="jd_uploader")
    if jd_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(jd_file.name)[1]) as tmp_jd:
            tmp_jd.write(jd_file.getbuffer())
            jd_text = screener.extract_text_from_file(tmp_jd.name)
        st.success(f"Loaded Job Description from `{jd_file.name}`")

with col2:
    st.subheader("2. Upload Any Resumes (PDF / TXT / DOCX)")
    uploaded_files = st.file_uploader(
        "Drag & drop one or more resumes here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="resume_uploader"
    )
    use_sample = st.checkbox("Also include sample resumes (optional, for instant demo testing)", value=False)

if st.button("🚀 Screen & Rank Uploaded Resumes", type="primary", use_container_width=True):
    temp_dir = tempfile.mkdtemp()
    resume_paths = []

    if uploaded_files:
        for uf in uploaded_files:
            file_path = os.path.join(temp_dir, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
            resume_paths.append(file_path)

    if use_sample:
        default_jd_path, sample_dir = generate_sample_dataset("sample_data")
        if not jd_text.strip():
            with open(default_jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
        for f in sorted(os.listdir(sample_dir)):
            if f.endswith((".pdf", ".txt")) and f != "job_description.txt":
                resume_paths.append(os.path.join(sample_dir, f))

    if not resume_paths:
        st.error("Please upload at least one resume (.pdf, .txt, or .docx) or check 'Also include sample resumes'.")
        st.stop()

    if not jd_text.strip():
        st.error("Please paste or upload a Job Description to match the resumes against.")
        st.stop()

    candidates, jd_skills = screener.screen_and_rank(jd_text, resume_paths)

    if not candidates:
        st.error("Could not extract readable text from the uploaded file(s). Please ensure the PDF is not a scanned image.")
        st.stop()

    st.divider()
    st.subheader(f"🏆 Candidate Ranking Leaderboard ({len(candidates)} Resumes Screened)")

    df = pd.DataFrame([
        {
            "Rank": c["rank"],
            "Candidate Name": c["name"],
            "Resume File": c["file"],
            "Overall Match Score (%)": c["overall_match_score"],
            "TF-IDF Similarity (%)": c["tfidf_similarity"],
            "Skill Match (%)": c["skill_match_score"],
            "Experience (Years)": c["experience_years"],
            "Education": c["education"],
            "Matched JD Skills": ", ".join(c["matched_skills"]),
            "All Extracted Skills": ", ".join(c["skills"]),
        }
        for c in candidates
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)

    df.to_csv("candidate_ranking_report.csv", index=False)
    st.download_button(
        label="📥 Download Candidate Ranking Report (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="candidate_ranking_report.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("📋 Detailed Candidate Summary Reports")
    if jd_skills:
        st.caption(f"**Target Skills Detected in Job Description ({len(jd_skills)}):** {', '.join(jd_skills)}")

    for c in candidates:
        with st.expander(f"Rank #{c['rank']}: {c['name']} — Match Score: {c['overall_match_score']:.2f}% ({c['file']})", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Match Score", f"{c['overall_match_score']:.2f}%")
            m2.metric("TF-IDF Similarity", f"{c['tfidf_similarity']:.2f}%")
            m3.metric("JD Skill Match", f"{c['skill_match_score']:.2f}%")
            m4.metric("Experience", f"{c['experience_years']} Years")

            st.write(f"**🎓 Education:** {c['education']}")
            st.write(f"**💼 Experience / Projects Summary:** {c['experience_summary']}")
            st.write(f"**✅ Matched JD Skills ({len(c['matched_skills'])}):** {', '.join(c['matched_skills']) if c['matched_skills'] else 'None'}")
            st.write(f"**🛠️ All Extracted Resume Skills ({len(c['skills'])}):** {', '.join(c['skills']) if c['skills'] else 'None'}")