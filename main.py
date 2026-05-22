import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama

st.set_page_config(page_title="AI PDF Chatbot")

st.title("AI PDF Chatbot")
st.write("Chat with your PDF using AI")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("PDF Loaded Successfully")

    user_question = st.text_input("Ask a question from your PDF")

    if user_question:

        relevant_docs = vectorstore.similarity_search(user_question)

        llm = Ollama(model="llama3")

        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain.run(
            input_documents=relevant_docs,
            question=user_question
        )

        st.write("### AI Response")
        st.write(response)
