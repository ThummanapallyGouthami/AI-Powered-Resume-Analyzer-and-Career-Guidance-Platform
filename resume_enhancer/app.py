import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
import google.generativeai as genai
import json
from difflib import get_close_matches

# --- Predefined Data for Popular Roles ---
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
    }
}

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_items_from_text(text, item_list):
    found_items = []
    for item in item_list:
        pattern = r"\b" + re.escape(item) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            found_items.append(item)
        else:
            close_matches = get_close_matches(item, text.split(), n=1, cutoff=0.8)
            if close_matches:
                found_items.append(item)
    return found_items

def fetch_data_for_custom_role(role):
    prompt = f"List 5 important skills, tools, and certifications for a {role}."
    genai.configure(api_key="your_api_key_here")  
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    text = response.text

    # Extract skills, tools, and certifications from the text
    skills = re.findall(r"Skills\s*:\s*(.*)", text)
    tools = re.findall(r"Tools\s*:\s*(.*)", text)
    certs = re.findall(r"Certifications\s*:\s*(.*)", text)

    return {
        "skills": skills[0].split(",") if skills else [],
        "tools": tools[0].split(",") if tools else [],
        "certifications": certs[0].split(",") if certs else []
    }

# --- Main UI Function ---
def main():
    st.set_page_config(page_title="Resume Enhancer", layout="centered")
    st.title("📄 AI-Powered Resume Enhancer")

    # --- Select Job Role ---
    st.subheader("1. Select or Enter a Job Role")
    col1, col2 = st.columns(2)
    with col1:
        predefined_role = st.selectbox(
            "Select a popular job role:",
            list(JOB_ROLE_DATA.keys()),
            index=None,
            placeholder="Choose from the list..."
        )
    with col2:
        custom_role = st.text_input(
            "Or enter a custom job role:",
            placeholder="e.g., AI Engineer"
        )

    selected_role = custom_role if custom_role else predefined_role

    # --- Upload Resume ---
    st.subheader("2. Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    # --- Run Analysis ---
    if uploaded_file and selected_role:
        st.markdown("---")
        st.subheader(f"Analyzing your resume for the **{selected_role}** role...")

        resume_text = extract_text_from_pdf(uploaded_file)

        # Get job role data
        if selected_role in JOB_ROLE_DATA:
            job_data = JOB_ROLE_DATA[selected_role]
        else:
            job_data = fetch_data_for_custom_role(selected_role)

        # Extracted Info
        extracted_skills = extract_items_from_text(resume_text, job_data["skills"])
        extracted_tools = extract_items_from_text(resume_text, job_data["tools"])
        extracted_certs = extract_items_from_text(resume_text, job_data["certifications"])

        # --- Display Extracted Data ---
        st.subheader("✅ Extracted Information")
        st.write("**Skills Found:**")
        st.write(", ".join(extracted_skills) if extracted_skills else "No known skills found.")
        st.write("**Tools Found:**")
        st.write(", ".join(extracted_tools) if extracted_tools else "No known tools found.")
        # Optionally enable this:
        # st.write("**Certifications Found:**")
        # st.write(", ".join(extracted_certs) if extracted_certs else "No known certifications found.")

        # --- Visual Skill Match ---
        st.subheader("📊 Skill Match Visualization")
        skill_data = [{"Skill": skill, "Present": 1 if skill in extracted_skills else 0} for skill in job_data["skills"]]
        df_skills = pd.DataFrame(skill_data)
        st.bar_chart(df_skills.set_index("Skill"))

        # --- Suggestions ---
        st.subheader("📌 Suggestions to Improve Your Resume")
        missing_skills = [s for s in job_data["skills"] if s not in extracted_skills]
        missing_tools = [t for t in job_data["tools"] if t not in extracted_tools]
        missing_certs = [c for c in job_data["certifications"] if c not in extracted_certs]

        if missing_skills or missing_tools or missing_certs:
            st.warning("Consider adding or emphasizing the following to match the job requirements:")
            if missing_skills:
                st.write(f"**Missing Skills:** {', '.join(missing_skills)}")
            if missing_tools:
                st.write(f"**Missing Tools:** {', '.join(missing_tools)}")
            if missing_certs:
                st.write(f"**Missing Certifications:** {', '.join(missing_certs)}")
        else:
            st.success("✅ Your resume includes all key skills, tools, and certifications for this role.")

# --- Run main if script executed directly ---
if __name__ == "__main__":
    main()
