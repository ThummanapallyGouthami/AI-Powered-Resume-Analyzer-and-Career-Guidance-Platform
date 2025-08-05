# resume_analyzer/analyzer.py
import streamlit as st
from .app_clean import main as run_analyzer_logic  # Assuming app_clean.py has a main() function

def analyzer_ui():
    st.title("📊 Resume Analyzer")
    run_analyzer_logic()
