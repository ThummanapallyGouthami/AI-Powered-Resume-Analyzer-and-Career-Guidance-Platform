import streamlit as st
import google.generativeai as genai
import textwrap
import re

# 🔐 Gemini API key
GEMINI_API_KEY = ""

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"❌ Failed to configure Gemini: {str(e)}")
    st.stop()

# 🌐 Streamlit App
def chatbot_ui():
    st.set_page_config(page_title="Interview Chatbot", layout="centered")

    st.markdown("""
    <h1 style='text-align: center; color: #1F4E79; font-size: 2.5rem;'>
        🎯 AI-Powered Career Chatbot
    </h1>
    <p style='text-align: center; font-size: 1.1rem;'>
        Get interview questions and answers based on your desired job role!
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # 👔 Select Job Role
    st.subheader("🧑‍💼 Select a Job Role")
    roles = ["Data Scientist", "Software Engineer", "Web Developer", "ML Engineer", "Business Analyst"]
    selected_role = st.selectbox("Choose your job role", roles)

    if st.button("🎯 Generate Mock Interview Questions"):
        if selected_role:
            with st.spinner("Generating interview questions..."):
                try:
                    prompt = f"""
                    Generate 10 technical interview questions for a {selected_role}.
                    Only list the questions in numbered format.
                    """
                    response = model.generate_content(prompt).text
                    st.subheader("🧠 Mock Interview Questions:")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please select a job role.")

    st.divider()

    # 💬 FAQ Buttons
    st.subheader("💡 Common Interview Questions")
    faq_list = [
        "What are the most important skills for a software engineer?",
        "How to answer 'Tell me about yourself'?",
        "What are some common SQL interview questions?",
        "What should I include in a data science resume?",
        "How do I prepare for a coding interview?"
    ]

    cols = st.columns(2)
    for i, faq in enumerate(faq_list):
        if cols[i % 2].button(faq, use_container_width=True):
            st.session_state.question = faq

    st.divider()

    # 📝 Custom Question Input
    st.subheader("📝 Ask Anything About Interviews or Careers")
    user_query = st.text_area("Type your question here:", value=st.session_state.get("question", ""), height=120)

    if st.button("💡 Get Answer", use_container_width=True):
        if user_query.strip():
            with st.spinner("Thinking..."):
                try:
                    answer_prompt = f"Answer the following career or interview question in detail: {user_query}"
                    answer = model.generate_content(answer_prompt).text
                    st.success(answer)
                except Exception as e:
                    st.error(f"Failed to get response: {e}")
        else:
            st.warning("Please enter a valid question.")

if __name__ == "__main__":
    chatbot_ui()
