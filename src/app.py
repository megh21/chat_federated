import streamlit as st
from pathlib import Path
from pdf_processing import PDFProcessor
from chat import ChatInterface
import os

class PDFChatApp:
    def __init__(self):
        # Initialize session state for navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'upload'
            
        # Initialize other session states if needed
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            
        # Create databases directory
        self.db_dir = Path("databases")
        self.db_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.chat_interface = ChatInterface()

    def run(self):
        """Main method to run the application"""
        # Sidebar navigation
        self._show_sidebar()
        
        # Display current page
        if st.session_state.current_page == 'upload':
            self._show_upload_page()
        elif st.session_state.current_page == 'chat':
            self._show_chat_page()

    def _show_sidebar(self):
        """Display sidebar with navigation"""
        with st.sidebar:
            st.title("PDF Chat Navigation")
            
            # Navigation buttons
            if st.button("📤 Upload PDFs", use_container_width=True):
                # Reset relevant session states when switching to upload
                st.session_state.current_page = 'upload'
                if 'processing_state' in st.session_state:
                    st.session_state.processing_state = 'idle'
                st.experimental_rerun()
                
            if st.button("💬 Chat Interface", use_container_width=True):
                # Reset relevant session states when switching to chat
                st.session_state.current_page = 'chat'
                if 'messages' in st.session_state:
                    st.session_state.messages = []
                st.experimental_rerun()
            
            # Show currently processed files if any
            if 'processed_files' in st.session_state and st.session_state.processed_files:
                st.divider()
                st.subheader("Processed Files")
                for file in st.session_state.processed_files:
                    st.write(f"📄 {file}")
            
            # Show current database if selected
            if 'retriever' in st.session_state and st.session_state.retriever:
                st.divider()
                st.subheader("Active Database")
                if hasattr(st.session_state, 'database_selector'):
                    st.write(f"📚 {st.session_state.database_selector}")

    def _show_upload_page(self):
        """Display PDF upload and processing page"""
        self.pdf_processor.process_pdfs()

    def _show_chat_page(self):
        """Display chat interface page"""
        self.chat_interface.run()

def main():
    # Check for API key
    if "COHERE_API_KEY" not in st.secrets or os.environ.get("COHERE_API_KEY") is None:
        st.error("Please set the Cohere API key in your Streamlit secrets")
        st.stop()
    
    # Initialize and run application
    app = PDFChatApp()
    app.run()

if __name__ == "__main__":
    main()