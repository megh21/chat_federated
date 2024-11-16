import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

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
            cohere_api_key= os.getenv("COHERE_API_KEY"),
            user_agent="my_app/1.0"
        )

        if 'chunk_size' not in st.session_state:
            st.session_state.chunk_size = 1000
        if 'chunk_overlap' not in st.session_state:
            st.session_state.chunk_overlap = 200
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )

    def _clean_text(self, text):
        """Clean the extracted text"""
        # Remove headers, footers, and page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Remove special characters and non-ASCII content
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # # Remove multiple spaces
        # text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        # text = re.sub(r'\n+', '\n', text)
        # Remove URLs
        # text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove references (e.g., [1], [2,3], [4-6])
        text = re.sub(r'\[\d+(,\s*\d+)*(-\d+)?\]', '', text)
        
        # Remove equations (e.g., (1), (2), (3))
        text = re.sub(r'\(\d+\)', '', text)
               
        # Remove figure and table captions (e.g., Figure 1:, Table 1:)
        text = re.sub(r'(Figure|Table) \d+:', '', text)
        
        return text.strip()

    def process_pdfs(self):
        """Main function to handle PDF upload and processing"""
        st.title("PDF to Vector Database Converter")
        
        # Add chunk size and overlap controls
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.chunk_size = st.number_input(
                "Chunk Size", 
                min_value=100, 
                max_value=2000, 
                value=st.session_state.chunk_size,
                help="Number of characters per chunk"
            )
        with col2:
            st.session_state.chunk_overlap = st.number_input(
                "Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=st.session_state.chunk_overlap,
                help="Number of characters to overlap between chunks"
            )

        # Update text splitter with new values
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )


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
                temp_path = Path(file.name)
                temp_path.write_bytes(file.getvalue())
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(str(temp_path))
                    docs = loader.load()
                    
                    # Clean and process each page
                    for doc in docs:
                        doc.page_content = self._clean_text(doc.page_content)
                    
                    # Split documents into chunks
                    chunked_docs = self.text_splitter.split_documents(docs)
                    
                    # Add source metadata
                    for doc in chunked_docs:
                        doc.metadata['source_file'] = file.name
                        if 'page' not in doc.metadata:
                            doc.metadata['page'] = 'unknown'
                    
                    documents.extend(chunked_docs)
                    if file.name not in st.session_state.processed_files:
                        st.session_state.processed_files.append(file.name)
                    st.success(f"Successfully processed: {file.name}")
                    
                    # Display processing stats
                    st.info(f"""
                    Processing stats for {file.name}:
                    - Original pages: {len(docs)}
                    - Chunks created: {len(chunked_docs)}
                    - Average chunk size: {sum(len(d.page_content) for d in chunked_docs) // len(chunked_docs)} characters
                    """)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                finally:
                    # temp_path.unlink(missing_ok=True)
                    pass
                    
        return documents
    

    def _handle_vector_store_creation(self):
        """Handle vector store creation and saving"""
        db_name = st.text_input("Enter database name", key="db_name_input")
        
        if db_name and st.button("Save Database", key="save_button"):
            try:
                with st.spinner("Creating vector store..."):
                    # Create vector store with metadata filtering
                    vector_store = FAISS.from_documents(
                        st.session_state.documents,
                        self.embeddings,
                        # dimensions=2048  # Set dimensions to 2048 for Cohere embeddings
                        
                        # metadata={"source_file": "str", "page": "str"}
                    )
                    
                    # Clean database name and save
                    db_name = "".join(c for c in db_name if c.isalnum())
                    save_path = self.db_dir / db_name
                    
                    # Save vector store
                    vector_store.save_local(str(save_path))
                    st.session_state.vector_store = vector_store
                    
                    # Add processing stats to success message
                    total_chunks = len(st.session_state.documents)
                    avg_chunk_size = sum(len(d.page_content) for d in st.session_state.documents) // total_chunks
                    
                    st.success(f"""
                    Database saved successfully!
                    - Name: {db_name}
                    - Location: {save_path}
                    - Files processed: {len(st.session_state.processed_files)}
                    - Total chunks: {total_chunks}
                    - Average chunk size: {avg_chunk_size} characters
                    - Chunk overlap: {st.session_state.chunk_overlap} characters
                    """)
                    
                    # Reset state
                    st.session_state.processing_state = 'idle'
                    st.session_state.documents = []
                    
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")

def main():
    # Check for API key
    if os.getenv("COHERE_API_KEY") is None:
        st.error("Please set the Cohere API key in your Streamlit secrets")
        st.stop()
    
    # Initialize and run processor
    processor = PDFProcessor()
    processor.process_pdfs()

if __name__ == "__main__":
    main()