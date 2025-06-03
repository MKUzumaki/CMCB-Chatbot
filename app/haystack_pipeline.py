from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor, EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

document_store = InMemoryDocumentStore(use_bm25=True)

# Load & convert PDF
converter = PDFToTextConverter(remove_numeric_tables=True)
docs = converter.convert(file_path="data/your-pdf.pdf", meta=None)

# Preprocess
preprocessor = PreProcessor(split_length=100, split_overlap=10)
processed_docs = preprocessor.process(docs)
document_store.write_documents(processed_docs)

# Retriever & Reader
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Final QA pipeline
pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)
