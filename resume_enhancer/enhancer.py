# resume_enhancer/enhancer.py
import streamlit as st
from .app import main as run_enhancer_logic  # Assuming app.py has a main() function

def enhancer_ui():
    st.title("🛠️ Resume Enhancer")
    run_enhancer_logic()
