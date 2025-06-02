from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor, EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import FARMReader

# 1. Create a Document Store
document_store = InMemoryDocumentStore(use_bm25=True)

# 2. Convert and Preprocess Documents
converter = PDFToTextConverter(remove_numeric_tables=True)
docs = converter.convert(file_path="data/1_National_Bank_of_Cambodia_TCRMG_Draft_V1_8_Consolidated_210225 (1).pdf", meta=None)

preprocessor = PreProcessor(split_length=100, split_overlap=10)
processed_docs = preprocessor.process(docs)
document_store.write_documents(processed_docs)

# 3. Retriever and Reader
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 4. Build QA Pipeline
pipe = ExtractiveQAPipeline(reader, retriever)
