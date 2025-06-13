import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vector_store")

    print(f"✅ Vector store saved with {len(documents)} chunks.")

if __name__ == "__main__":
    create_vector_store("data/scraped_data.txt")
