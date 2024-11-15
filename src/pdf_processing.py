# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_cohere import CohereEmbeddings
# import os
# from dotenv import load_dotenv
# import time
# load_dotenv()

# cohere_api_key = os.getenv("COHERE_API_KEY")
# if not cohere_api_key:
#     st.error("Cohere API Key not found! Please set the 'COHERE_API_KEY' environment variable.")
#     st.stop()

# # Function to handle file uploads and process them
# def handle_file_upload():
#     uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
#     if st.button("process"):
#         if uploaded_files:
#             with st.spinner("Processing files..."):
#                 # Collect all documents from the uploaded PDFs
#                 all_documents = []
                
#                 # Process each uploaded file
#                 for uploaded_file in uploaded_files:
#                     with open(uploaded_file.name, "wb") as f:
#                         f.write(uploaded_file.getbuffer())
                    
#                     # Extract documents from the uploaded PDF
#                     documents = convert_pdf_to_documents(uploaded_file.name)
#                     all_documents.extend(documents)
                
#                 # Create a single vector store from all documents
#                 create_vector_store(all_documents)
#                 st.success(f"All files combined into a single vector store successfully!")

# # Convert a single PDF to documents (text extraction)
# def convert_pdf_to_documents(file_name):
#     loader = PyPDFLoader(file_name)
#     documents = loader.load()
#     return documents

# # Function to create a vector store from a list of documents
# def create_vector_store(documents):
#     # Using Cohere embeddings for text processing
#     embedding = CohereEmbeddings(
#         model="embed-english-light-v3.0",
#         cohere_api_key=cohere_api_key,
#         user_agent="my_app/1.0"  # Specify user_agent manually
#     )
    
#     # Create a FAISS vector store from the documents
#     vector_store = FAISS.from_documents(documents, embedding)
    
#     # Ensure the 'databases' directory exists
#     if not os.path.exists("databases"):
#         os.makedirs("databases")
    
#     # Take user input for the name of the vector store
#     vector_store_name = st.text_input("Enter the name of the vector store")
    
#     # Remove any special characters from the name
#     vector_store_name = "".join(e for e in vector_store_name if e.isalnum())
    
#     # If name is not provided, use the default name with the timestamp
#     if not vector_store_name:
#         vector_store_name = "vectorstore({:.0f})".format(time.time())
    
#     # Button to finish and save the database
#     if vector_store_name:
#         if st.button("Finish and Save Database"):
#             # Save the vector store to the 'databases' folder
#             vector_store.save_local(folder_path="databases/" + vector_store_name)
#             st.success("Vector store created and saved successfully!")

# # Main function to run the app
# def main():
#     st.title("Create New Database with PDFs")
    
#     # Call the function to handle PDF uploads
#     handle_file_upload()

# if __name__ == "__main__":
#     main()

## ***********The second try of the code**********



# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_cohere import CohereEmbeddings
# import os
# from dotenv import load_dotenv
# import time
# load_dotenv()

# cohere_api_key = os.getenv("COHERE_API_KEY")
# if not cohere_api_key:
#     st.error("Cohere API Key not found! Please set the 'COHERE_API_KEY' environment variable.")
#     st.stop()

# # Function to handle file uploads and process them
# def handle_file_upload():
#     uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
#     if st.button("Process Files", key="process_files_button"):
#         if uploaded_files:
#             with st.spinner("Processing files..."):
#                 # Collect all documents from the uploaded PDFs
#                 all_documents = []
                
#                 # Process each uploaded file
#                 for uploaded_file in uploaded_files:
#                     with open(uploaded_file.name, "wb") as f:
#                         f.write(uploaded_file.getbuffer())
                    
#                     # Extract documents from the uploaded PDF
#                     documents = convert_pdf_to_documents(uploaded_file.name)
#                     all_documents.extend(documents)
                
#                 # Create a single vector store from all documents
#                 create_vector_store(all_documents)

# # Convert a single PDF to documents (text extraction)
# def convert_pdf_to_documents(file_name):
#     loader = PyPDFLoader(file_name)
#     documents = loader.load()
#     return documents

# # Function to create a vector store from a list of documents
# def create_vector_store(documents):
#     # Using Cohere embeddings for text processing
#     embedding = CohereEmbeddings(
#         model="embed-english-light-v3.0",
#         cohere_api_key=cohere_api_key,
#         user_agent="my_app/1.0"  # Specify user_agent manually
#     )
    
#     # Create a FAISS vector store from the documents
#     vector_store = FAISS.from_documents(documents, embedding)
    
#     # Ensure the 'databases' directory exists
#     if not os.path.exists("databases"):
#         os.makedirs("databases")
    
#     def save_database_callback():
#         if vector_store_name:
#             # Save the vector store to the 'databases' folder
#             vector_store.save_local(folder_path="databases/" + vector_store_name)
#             st.success("Vector store created and saved successfully!")
#         else:
#             st.error("Please provide a name for the vector store.")

#     # Take user input for the name of the vector store
#     vector_store_name = st.text_input("Enter the name of the vector store", key="vector_store_name_input")
    
#     # Remove any special characters from the name
#     vector_store_name = "".join(e for e in vector_store_name if e.isalnum())
    
#     # If name is empty, use the default name with the timestamp
#     if not vector_store_name:
#         print("No name provided, using default name.")
#         vector_store_name = "vectorstore({:.0f})".format(time.time())

#     # Button to finish and save the database
#     if vector_store_name:
#        if st.button("Finish and Save Database", key="finish_save_database_button"):
#           save_database_callback()
        

# # Main function to run the app
# def main():
#     st.title("Create New Database with PDFs")
    
#     # Call the function to handle PDF uploads
#     handle_file_upload()

# if __name__ == "__main__":
#     main()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self):
        # Initialize session state variables
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'processing_state' not in st.session_state:
            st.session_state.processing_state = 'idle'  # States: idle, processing, saving
        if 'documents' not in st.session_state:
            st.session_state.documents = []
            
        # Create databases directory
        self.db_dir = Path("databases")
        self.db_dir.mkdir(exist_ok=True)
        
        # Initialize Cohere embeddings
        self.embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0",
            cohere_api_key=st.secrets["COHERE_API_KEY"],
            user_agent="my_app/1.0"
        )

    def process_pdfs(self):
        """Main function to handle PDF upload and processing"""
        st.title("PDF to Vector Database Converter")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDFs", 
            type="pdf", 
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        # Show current state
        st.write(f"Current state: {st.session_state.processing_state}")
        if st.session_state.processed_files:
            st.write("Processed files:", ", ".join(st.session_state.processed_files))

        if not uploaded_files:
            st.info("Please upload PDF files to begin")
            return

        # Process PDFs button
        if st.button("Process PDFs", key="process_button"):
            st.session_state.processing_state = 'processing'
            st.session_state.documents = self._load_documents(uploaded_files)
            if st.session_state.documents:
                st.session_state.processing_state = 'saving'
                st.experimental_rerun()

        # Show database name input and save button only after processing
        if st.session_state.processing_state == 'saving' and st.session_state.documents:
            self._handle_vector_store_creation()

    def _load_documents(self, uploaded_files):
        """Load and process PDF documents"""
        documents = []
        with st.spinner("Processing PDFs..."):
            for file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = Path(file.name)
                temp_path.write_bytes(file.getvalue())
                
                try:
                    # Load and process PDF
                    loader = PyPDFLoader(str(temp_path))
                    current_docs = loader.load()
                    documents.extend(current_docs)
                    if file.name not in st.session_state.processed_files:
                        st.session_state.processed_files.append(file.name)
                    st.success(f"Successfully processed: {file.name}")
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                finally:
                    # Cleanup temporary file
                    temp_path.unlink(missing_ok=True)
                    
        return documents

    def _handle_vector_store_creation(self):
        """Handle vector store creation and saving"""
        st.write("Creating vector store...")
        
        # Get database name
        db_name = st.text_input(
            "Enter database name:",
            key="db_name",
            help="Enter a name for your vector database"
        )
        
        # Save button
        if db_name and st.button("Save Database", key="save_button"):
            try:
                with st.spinner("Creating vector store..."):
                    # Create vector store
                    vector_store = FAISS.from_documents(
                        st.session_state.documents, 
                        self.embeddings
                    )
                    
                    # Clean database name and save
                    db_name = "".join(c for c in db_name if c.isalnum())
                    save_path = self.db_dir / db_name
                    
                    # Save vector store
                    vector_store.save_local(str(save_path))
                    st.session_state.vector_store = vector_store
                    
                    # Success message
                    st.success(f"""
                    Database saved successfully!
                    - Name: {db_name}
                    - Location: {save_path}
                    - Files processed: {len(st.session_state.processed_files)}
                    """)
                    
                    # Reset state
                    st.session_state.processing_state = 'idle'
                    st.session_state.documents = []
                    
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")

def main():
    # Check for API key
    if "COHERE_API_KEY" not in st.secrets:
        st.error("Please set the Cohere API key in your Streamlit secrets")
        st.stop()
    
    # Initialize and run processor
    processor = PDFProcessor()
    processor.process_pdfs()

if __name__ == "__main__":
    main()