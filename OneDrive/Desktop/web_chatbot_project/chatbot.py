from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def answer_query(query: str) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector_store", embeddings,allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=3)

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    return answer

if __name__ == "__main__":
    print("🤖 Welcome to your AI Chatbot! Type 'exit' to quit.\n")
    while True:
        query = input("💬 You: ")
        if query.lower() in ("exit", "quit"):
            print("👋 Chatbot session ended.")
            break
        response = answer_query(query)
        print("🤖 Bot:", response, "\n")
