# web_ui/streamlit_app.py

import streamlit as st
from haystack_pipeline.py import pipe

st.set_page_config(page_title="Cyber Policy Chatbot", layout="centered")
st.title("📘 Cybersecurity Policy Chatbot")
st.write("Ask a question based on the uploaded documents.")

query = st.text_input("🔎 Enter your question:")
if query:
    with st.spinner("Searching for answers..."):
        result = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
        if result["answers"]:
            st.success(f"🗨️ Answer: {result['answers'][0].answer}")
        else:
            st.warning("Sorry, I couldn't find an answer.")
