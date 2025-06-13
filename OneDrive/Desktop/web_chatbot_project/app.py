import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -------------- PAGE SETUP ------------------
st.set_page_config(page_title="Inovabeing AI Chatbot", layout="centered")

# Custom styles
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .title {
            color: #003366;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #444444;
            font-size: 18px;
        }
        .bot-box {
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 5px solid #3399ff;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 16px;
            color: #000;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# -------------- HEADER ------------------
st.markdown('<div class="title">🤖 Inovabeing AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Built by Ravikiran | Powered by Hugging Face</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------- LOAD VECTOR STORE + CHAIN ------------------
@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    similarity_search = vectorstore.similarity_search

    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=qa_model)

    chain = load_qa_chain(llm, chain_type="stuff")
    return similarity_search, chain

# -------------- RECOMMENDED QUESTIONS ------------------
recommended = [
    "What does Inovabeing do?",
    "What services are offered?",
    "Who can use this chatbot?",
    "Is this chatbot powered by OpenAI?",
    "What technologies are used in the project?"
]

st.markdown("### 💡 Try a recommended question:")

cols = st.columns(2)
for i, q in enumerate(recommended):
    if cols[i % 2].button(q):
        st.session_state.selected_query = q

query = st.text_input("💬 Or ask your own question:", value=st.session_state.get("selected_query", ""))

if query and "selected_query" in st.session_state:
    del st.session_state["selected_query"]


if query:
    with st.spinner("Generating smart response..."):
        similarity_search, chain = load_chain()
        top_docs = similarity_search(query, k=3)
        answer = chain.run(input_documents=top_docs, question=query)

        # Display styled answer
        st.markdown(
            f"<div class='bot-box'><b>🤖 Answer:</b><br>{answer}</div>",
            unsafe_allow_html=True
        )
