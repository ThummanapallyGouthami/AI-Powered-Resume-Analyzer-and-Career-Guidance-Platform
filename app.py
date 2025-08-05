# app.py
import streamlit as st
from login import login_page
from home import home
from chatbot.app import chatbot_ui
from resume_analyzer.analyzer import analyzer_ui
from resume_enhancer.enhancer import enhancer_ui

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Routing logic
if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("📂 Navigation")
    choice = st.sidebar.radio("Go to", ["Home", "Resume Analyzer", "Resume Enhancer", "ChatBot"])

    if choice == "Home":
        home()
    elif choice == "Resume Analyzer":
        analyzer_ui()
    elif choice == "Resume Enhancer":
        enhancer_ui()
    elif choice == "ChatBot":
        chatbot_ui()
