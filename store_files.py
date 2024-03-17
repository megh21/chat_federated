import os
import gc
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings

load_dotenv()

def download_file(url):
    """
    Download a file from a given URL and save it as 'downloaded_file.pdf'.
    """
    if url:
        response = requests.get(url)
        with open('downloaded_file.pdf', 'wb') as f:
            f.write(response.content)
        st.success("File downloaded successfully!")
        return 'downloaded_file.pdf'
    return None

def display_vector_stores():
    """
    Display the vector stores in the 'store' directory.
    """
    vector_stores = os.listdir("store")
    if vector_stores:
        df = pd.DataFrame(vector_stores, columns=["Vector Store"])
        df["Created at"] = [datetime.fromtimestamp(
            os.path.getctime(os.path.join("store", store))
            ).strftime('%Y-%m-%d %H:%M:%S') for store in vector_stores]
    else:
        df = pd.DataFrame(columns=["Vector Store", "Created at"])
    st.table(df)

def convert_to_vector_store(uploaded_file):
    """
    Convert an uploaded file to a vector store.
    """
    with open("temp_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp_file.pdf")
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    page = loader.load()
    split_text = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )
    split_pages = split_text.split_documents(page)
    db = FAISS.from_documents(split_pages, embedding=embedding, allow_dangerous_deserialization=True)

    vector_stores = os.listdir("store")
    display_vector_stores()

    if st.button("Create new"):
        save_path = os.path.join("store", "vectorstore({:.0f})".format(len(vector_stores)+1))
        db.save_local(folder_path=save_path)
        st.success(f"New Vector store saved successfully at {save_path}")
    elif st.button("Merge"):
        selected_store = st.selectbox("Select a vector store to merge", vector_stores)
        save_path = os.path.join("store", selected_store)
        local_db = FAISS.load_local(save_path, embedding,allow_dangerous_deserialization=True)
        local_db.merge_from(db)
        st.write("Merge completed")
        local_db.save_local(save_path)
        st.success(f"Vector store updated successfully at {save_path}")

    for store in vector_stores:
        if st.button(f"Delete {store}"):
            os.remove(os.path.join("store", store))
            st.success(f"Vector store {store} deleted successfully")

    os.remove("temp_file.pdf")
    _ = gc.collect()

def file_converter_page():
    """
    Main function for the file converter page.
    """
    st.title("File to Vector Store Converter")
    uploaded_files = st.file_uploader("Drag and drop or select files", type="pdf", accept_multiple_files=True)
    url = st.text_input("Enter a URL to download a document", "https://example.com")
    
    if st.button("Download"):
        uploaded_file = download_file(url)
        if uploaded_file:
            st.write("Converting file to vector store...")
            convert_to_vector_store(uploaded_file)
    
    display_vector_stores()

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            st.write("Converting file to vector store...")
            convert_to_vector_store(uploaded_file)

        st.sidebar.markdown("## Navigation")
        if st.sidebar.button("Go to Chatbot"):
            st.rerun()

if __name__ == "__main__":
    file_converter_page()