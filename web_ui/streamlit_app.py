import streamlit as st
from app.haystack_pipeline import pipe

st.title("📄 Chat with your PDF")

query = st.text_input("Ask a question about the document:")

if query:
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    answer = prediction['answers'][0].answer
    st.write("Answer:", answer)
