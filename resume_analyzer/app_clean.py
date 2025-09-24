import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

# --- Load Sentence-BERT model once ---
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# --- Expanded predefined roles ---
JOB_ROLE_DATA = {
    "Data Scientist": {
        "skills": ["Python", "R", "SQL", "Machine Learning", "Deep Learning", "Statistics", "Data Visualization"],
        "tools": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Matplotlib"],
        "certifications": ["IBM Data Science Professional Certificate", "Google Data Analytics Certificate"]
    },
    "Web Developer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "REST APIs", "Responsive Design"],
        "tools": ["VS Code", "Chrome DevTools", "Git", "Webpack"],
        "certifications": ["Meta Front-End Developer", "freeCodeCamp Responsive Web Design"]
    },
    "AI Engineer": {
        "skills": ["Python", "TensorFlow", "PyTorch", "Computer Vision", "NLP", "Model Deployment"],
        "tools": ["TensorFlow", "PyTorch", "OpenCV", "Hugging Face Transformers"],
        "certifications": ["TensorFlow Developer Certificate", "AI For Everyone by Andrew Ng"]
    },
    "ML Engineer": {
        "skills": ["Python", "Scikit-learn", "Machine Learning", "Deep Learning", "Data Engineering"],
        "tools": ["TensorFlow", "PyTorch", "Keras", "Airflow", "Docker"],
        "certifications": ["AWS Machine Learning Specialty", "TensorFlow Developer Certificate"]
    },
    "Software Engineer": {
        "skills": ["Java", "Python", "C++", "Algorithms", "Data Structures"],
        "tools": ["Git", "VS Code", "Jira", "Docker"],
        "certifications": ["Oracle Java Certification", "AWS Certified Developer"]
    },
    "DevOps Engineer": {
        "skills": ["CI/CD", "Automation", "Cloud", "Docker", "Kubernetes"],
        "tools": ["Jenkins", "Docker", "Kubernetes", "Terraform", "Git"],
        "certifications": ["AWS DevOps Engineer", "Docker Certified Associate"]
    },
    "Cloud Engineer": {
        "skills": ["AWS", "Azure", "GCP", "Networking", "Cloud Architecture"],
        "tools": ["Terraform", "Ansible", "Kubernetes", "Docker"],
        "certifications": ["AWS Solutions Architect", "Google Cloud Professional"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Network Security", "Penetration Testing", "Incident Response", "Threat Analysis"],
        "tools": ["Wireshark", "Nmap", "Metasploit", "Splunk"],
        "certifications": ["CEH", "CISSP", "CompTIA Security+"]
    },
    "Business Analyst": {
        "skills": ["Requirement Analysis", "SQL", "Data Visualization", "Documentation"],
        "tools": ["Excel", "Power BI", "Tableau", "Jira"],
        "certifications": ["CBAP", "IIBA Certification"]
    },
    "Product Manager": {
        "skills": ["Roadmap Planning", "Market Research", "User Stories", "Stakeholder Management"],
        "tools": ["Jira", "Trello", "Aha!", "Miro"],
        "certifications": ["Certified Scrum Product Owner", "PMP"]
    },
    "QA Engineer": {
        "skills": ["Test Automation", "Manual Testing", "Selenium", "Performance Testing"],
        "tools": ["Selenium", "Jira", "Postman", "TestRail"],
        "certifications": ["ISTQB Foundation", "Certified Software Tester"]
    }
}

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()

def extract_items_from_text(text, item_list):
    found_items = []
    for item in item_list:
        pattern = r"\b" + re.escape(item.lower()) + r"\b"
        if re.search(pattern, text):
            found_items.append(item)
        else:
            close_matches = get_close_matches(item.lower(), text.split(), n=1, cutoff=0.8)
            if close_matches:
                found_items.append(item)
    return found_items

def calculate_match(extracted, required):
    matched = set(extracted) & set(required)
    return len(matched)/len(required)*100 if required else 0

def calculate_semantic_similarity(text1, text2):
    emb1 = MODEL.encode(text1)
    emb2 = MODEL.encode(text2)
    sim = util.cos_sim(emb1, emb2).item()
    return sim * 100

def top_3_suggestions(missing_skills, missing_tools, missing_certs):
    suggestions = missing_skills + missing_tools + missing_certs
    return suggestions[:3] if suggestions else ["No suggestions, resume is strong!"]

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    st.title("📄 AI-Powered Resume Analyzer & Advisor")

    # --- Only Predefined Role Selection ---
    selected_role = st.selectbox("Select a job role", list(JOB_ROLE_DATA.keys()))

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if uploaded_file and selected_role:
        resume_text = extract_text_from_pdf(uploaded_file)

        job_data = JOB_ROLE_DATA[selected_role]

        # --- Extract items ---
        extracted_skills = extract_items_from_text(resume_text, job_data["skills"])
        extracted_tools = extract_items_from_text(resume_text, job_data["tools"])
        extracted_certs = extract_items_from_text(resume_text, job_data["certifications"])

        # --- Match Scores ---
        skill_score = calculate_match(extracted_skills, job_data["skills"])
        tool_score = calculate_match(extracted_tools, job_data["tools"])
        cert_score = calculate_match(extracted_certs, job_data["certifications"])
        semantic_score = calculate_semantic_similarity(
            resume_text, " ".join(job_data["skills"] + job_data["tools"] + job_data["certifications"])
        )

        overall_score = 0.4*skill_score + 0.2*tool_score + 0.1*cert_score + 0.3*semantic_score

        st.subheader("📈 Match Scores")
        scores_df = pd.DataFrame({
            "Category": ["Skills", "Tools", "Certifications", "Semantic"],
            "Score (%)": [skill_score, tool_score, cert_score, semantic_score]
        })
        st.bar_chart(scores_df.set_index("Category"))
        st.success(f"💡 Overall Resume Match Score: {overall_score:.2f}%")

        # --- Suggestions ---
        missing_skills = [s for s in job_data["skills"] if s not in extracted_skills]
        missing_tools = [t for t in job_data["tools"] if t not in extracted_tools]
        missing_certs = [c for c in job_data["certifications"] if c not in extracted_certs]

        st.subheader("📌 Suggestions for Improvement")
        if missing_skills or missing_tools or missing_certs:
            if missing_skills: st.write(f"**Missing Skills:** {', '.join(missing_skills)}")
            if missing_tools: st.write(f"**Missing Tools:** {', '.join(missing_tools)}")
            if missing_certs: st.write(f"**Missing Certifications:** {', '.join(missing_certs)}")
        else:
            st.success("✅ Your resume covers all key requirements!")

        # --- Top 3 Recommendations ---
        st.subheader("💡 Top 3 Recommendations")
        for idx, suggestion in enumerate(top_3_suggestions(missing_skills, missing_tools, missing_certs), 1):
            st.write(f"{idx}. {suggestion}")

        # --- Resume Verdict ---
        st.subheader("📝 Resume Verdict / Hiring Suggestion")
        if overall_score >= 80:
            st.success("✅ Excellent! Your resume is likely to be accepted for this role.")
        elif overall_score >= 60:
            st.warning("⚠️ Moderate match. Improvements recommended.")
        else:
            st.error("❌ Weak match. Resume unlikely to be accepted.")

        # --- Radar Chart ---
        st.subheader("📊 Skill Coverage Radar")
        categories = job_data["skills"]
        values = [1 if skill in extracted_skills else 0 for skill in categories]
        values += values[:1]
        categories += categories[:1]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), categories)
        ax.set_ylim(0,1)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
