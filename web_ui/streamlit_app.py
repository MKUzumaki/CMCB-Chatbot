import streamlit as st
import fitz

from pathlib import Path

from haystack.document_stores import InMemoryDocumentStore
# from haystack.nodes import PDFToTextConverter, PreProcessor, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader
from haystack.schema import Document

# 1. Create a Document Store
document_store = InMemoryDocumentStore(use_bm25=True)

# 2. Convert and Preprocess Documents
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    returen text
    
pdf_path = Path(__file__).parent.parent / "data" / "TCRMG.pdf"
text = extract_text_from_pdf(str(pdf_path))
docs = [{"content": text}]

preprocessor = PreProcessor(split_length=100, split_overlap=10)
processed_docs = preprocessor.process(docs)
document_store.write_documents(processed_docs)

# 3. Retriever and Reader
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 4. Build QA Pipeline
pipe = ExtractiveQAPipeline(reader, retriever)

# web_ui/streamlit_app.py

#cut

st.set_page_config(page_title="Cyber Policy Chatbot", layout="centered")
st.title("📘 Cybersecurity Policy Chatbot")
st.write("Ask a question based on the uploaded documents.")

query = st.text_input("🔎 Enter your question:")
if query:
    with st.spinner("Searching for answers..."):
        result = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
        if result["answers"] and result["answers"][0].answer:
            st.success(f"🗨️ Answer: {result['answers'][0].answer}")
        else:
            st.warning("Sorry, I couldn't find a valid answer.")
