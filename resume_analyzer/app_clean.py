
import nltk

# Download NLTK data safely
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import pdfplumber
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

@st.cache_data
def load_data():
    return pd.read_csv("resume_analyzer\AI_Resume_Screening.csv")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def extract_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            text = docx2txt.process(file)
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    return text

def analyze_skills(resume_text, job_desc):
    resume_skills = set(preprocess_text(resume_text).split())
    job_skills = set(preprocess_text(job_desc).split())
    matching_skills = job_skills & resume_skills
    missing_skills = job_skills - resume_skills
    match_percentage = len(matching_skills) / len(job_skills) * 100 if job_skills else 0
    return {
        "matching_skills": list(matching_skills),
        "missing_skills": list(missing_skills),
        "match_percentage": match_percentage,
    }

def analyze_experience(resume_text, required_exp):
    exp_pattern = r'(\d+)\s*(years?|yrs?)'
    matches = re.findall(exp_pattern, resume_text.lower())
    resume_exp = max([int(m[0]) for m in matches]) if matches else 0
    return {
        "resume_experience": resume_exp,
        "required_experience": required_exp,
        "meets_requirement": resume_exp >= required_exp,
    }

def analyze_education(resume_text, required_degrees):
    degree_keywords = ['bachelor', 'master', 'phd', 'mba', 'bsc', 'msc', 'b.tech', 'm.tech']
    found_degrees = {word for word in degree_keywords if word in resume_text.lower()}
    required_degrees_lower = {deg.lower() for deg in required_degrees}
    return {
        "found_degrees": list(found_degrees),
        "required_degrees": required_degrees,
        "meets_requirement": any(deg in found_degrees for deg in required_degrees_lower),
    }

def analyze_certifications(resume_text, required_certs):
    required_certs = [cert for cert in required_certs if cert != 'None']
    if not required_certs:
        return {"found_certifications": [], "required_certifications": [], "meets_requirement": True}
    found_certs = [cert for cert in required_certs if cert.lower() in resume_text.lower()]
    return {
        "found_certifications": found_certs,
        "required_certifications": required_certs,
        "meets_requirement": len(found_certs) >= len(required_certs),
    }

def calculate_similarity(resume_text, job_desc):
    processed_resume = preprocess_text(resume_text)
    processed_job_desc = preprocess_text(job_desc)
    if not processed_resume or not processed_job_desc:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([processed_resume, processed_job_desc])
        return cosine_similarity(vectors[0], vectors[1])[0][0] * 100
    except ValueError:
        return 0.0

def generate_visualizations(analysis_results):
    plt.style.use('seaborn-v0_8-talk')
    figs = []

    skills_data = {
        "Matched": len(analysis_results["skills_analysis"]["matching_skills"]),
        "Missing": len(analysis_results["skills_analysis"]["missing_skills"])
    }
    fig, ax = plt.subplots()
    ax.bar(skills_data.keys(), skills_data.values(), color=['#2ecc71', '#e74c3c'])
    ax.set_title("Skills Match", fontsize=16)
    ax.set_ylabel("Count")
    figs.append(fig)

    exp_data = {
        "Resume": analysis_results["experience_analysis"]["resume_experience"],
        "Required": analysis_results["experience_analysis"]["required_experience"]
    }
    fig, ax = plt.subplots()
    ax.bar(exp_data.keys(), exp_data.values(), color=['#3498db', '#f39c12'])
    ax.set_title("Experience Comparison", fontsize=16)
    ax.set_ylabel("Years")
    figs.append(fig)

    return figs

def main():
    st.set_page_config(layout="wide", page_title="AI Resume Analyzer")
    st.title("📄 AI Resume Analyzer")

    st.write("Upload a resume and select a job role to see the match analysis.")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Error: `AI_Resume_Screening.csv` not found. Please upload it.")
        return

    with st.sidebar:
        st.header("Inputs")
        job_options = list(df['Job Role'].unique())
        job_title = st.selectbox("Select Job Role", job_options)
        sample_job = df[df['Job Role'] == job_title].iloc[0]

        job_desc_default = f"Required Skills: {sample_job.get('Skills', '')}\nRequired Experience: {sample_job.get('Experience (Years)', 0)} years\nRequired Education: {sample_job.get('Education', '')}"
        job_desc = st.text_area("Job Description", value=job_desc_default, height=200)

        uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx"])

        st.header("Refine Requirements")
        required_experience = st.slider("Required Experience (Years)", 0, 20, int(sample_job['Experience (Years)']))

        edu_opts = ['B.Sc', 'B.Tech', 'MBA', 'M.Tech', 'PhD']
        def_edu = [e for e in [sample_job['Education']] if e in edu_opts]
        required_degrees = st.multiselect("Required Education", edu_opts, default=def_edu)

        cert_opts = ['AWS Certified', 'Google ML', 'Deep Learning Specialization', 'None']
        def_certs = [c for c in [sample_job['Certifications']] if pd.notna(c) and c in cert_opts]
        required_certifications = st.multiselect("Required Certifications", cert_opts, default=def_certs)

    if st.sidebar.button("Analyze Resume", use_container_width=True):
        if uploaded_file:
            with st.spinner("Analyzing... please wait."):
                resume_text = extract_text_from_file(uploaded_file)
                if resume_text:
                    analysis = {
                        "skills_analysis": analyze_skills(resume_text, job_desc),
                        "experience_analysis": analyze_experience(resume_text, required_experience),
                        "education_analysis": analyze_education(resume_text, required_degrees),
                        "certification_analysis": analyze_certifications(resume_text, required_certifications),
                        "similarity_score": calculate_similarity(resume_text, job_desc),
                    }

                    st.header("Analysis Dashboard")
                    st.metric("Overall Match Score", f"{analysis['similarity_score']:.1f}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Visual Summary")
                        figs = generate_visualizations(analysis)
                        for fig in figs:
                            st.pyplot(fig)

                    with col2:
                        st.subheader("Requirements Check")
                        st.success(f"**Experience:** {'Met' if analysis['experience_analysis']['meets_requirement'] else 'Not Met'}")
                        st.success(f"**Education:** {'Met' if analysis['education_analysis']['meets_requirement'] else 'Not Met'}")
                        st.success(f"**Certifications:** {'Met' if analysis['certification_analysis']['meets_requirement'] else 'Not Met'}")

                        with st.expander("Skills Breakdown"):
                            st.write(f"**Matching Skills:** {', '.join(analysis['skills_analysis']['matching_skills'])}")
                            st.write(f"**Missing Skills:** {', '.join(analysis['skills_analysis']['missing_skills'])}")

                        with st.expander("Education Breakdown"):
                            st.write(f"**Required:** {', '.join(analysis['education_analysis']['required_degrees'])}")
                            st.write(f"**Found:** {', '.join(analysis['education_analysis']['found_degrees'])}")
                else:
                    st.error("Could not extract text from the resume.")
        else:
            st.warning("Please upload a resume before analyzing.")

if __name__ == '__main__':
    main()
