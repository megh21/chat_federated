import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere, CohereEmbeddings
from pathlib import Path
from typing import Dict, List, Tuple
import uuid

class ChatInterface:
    def __init__(self):
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        
        # Initialize Cohere components
        self.embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0",
            cohere_api_key=st.secrets["COHERE_API_KEY"]
        )
        self.chat_model = ChatCohere(
            model="command-r",
            temperature=0.3,  # Lower temperature for more focused responses
            cohere_api_key=st.secrets["COHERE_API_KEY"]
        )

    def run(self):
        """Main method to run the chat interface"""
        st.title("Chat with PDFs")
        
        # First, handle database selection
        if not st.session_state.retriever:
            self._setup_database()
        else:
            self._show_chat_interface()

    def _setup_database(self):
        """Handle database selection and initialization"""
        st.subheader("Select Database")
        
        # Check for databases directory
        db_dir = Path("databases")
        if not db_dir.exists():
            st.error("No databases directory found. Please create some databases first.")
            return
        
        # Get available databases
        databases = [d.name for d in db_dir.iterdir() if d.is_dir()]
        if not databases:
            st.error("No databases found. Please create a database first.")
            return
        
        # Database selection
        selected_db = st.selectbox(
            "Select a database to chat with:",
            databases,
            key="database_selector"
        )
        
        # Load database button
        if st.button("Load Database"):
            with st.spinner("Loading database..."):
                try:
                    vector_store = FAISS.load_local(
                        str(db_dir / selected_db),
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.session_state.retriever = vector_store.as_retriever(
                        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
                    )
                    st.success("Database loaded successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading database: {str(e)}")

    def _show_chat_interface(self):
        """Display the chat interface and handle messages"""
        st.subheader("Chat Interface")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"```\n{source}\n```")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = self._generate_response(prompt)
                    st.write(response)
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"```\n{source}\n```")
                    
                    # Save message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

    def _generate_response(self, question: str) -> Tuple[str, List[str]]:
        """Generate a response using RAG with source tracking"""
        try:
            # Get relevant documents
            docs = st.session_state.retriever.invoke(question)
            
            # Create a unique ID for each chunk for reference
            chunks_with_ids = []
            source_map = {}
            
            for doc in docs:
                chunk_id = str(uuid.uuid4())[:8]
                chunks_with_ids.append(f"[{chunk_id}] {doc.page_content}")
                source_map[chunk_id] = doc.page_content
            
            # Create context with identifiable chunks
            context = "\n\n".join(chunks_with_ids)
            
            # Create prompt
            prompt = f"""You are a helpful research assistant. Answer the following question based on the provided context.
            Use information ONLY from the provided context. If you're not sure or if the context doesn't contain the answer,
            say so. Reference the source IDs [xxxx] when using information from the context.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Answer the question using ONLY the provided context
            2. Reference source IDs [xxxx] when using specific information
            3. If you can't answer from the context, say so
            4. Be concise but thorough
            5. Use analytical and scientific language
            """

            # Get response from LLM
            response = self.chat_model.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            print('source_map) \n')
            print(source_map)
            # Extract used sources
            used_sources = []
            for chunk_id in source_map.keys():
                if chunk_id in response_text:
                    used_sources.append(source_map[chunk_id])
            
            # Clean up response by removing source IDs for display
            cleaned_response = response_text
            for chunk_id in source_map.keys():
                cleaned_response = cleaned_response.replace(f"[{chunk_id}]", "")
            print("used_sources: \n")
            print(used_sources)
            return cleaned_response.strip(), used_sources
            
        except Exception as e:
            return f"Error generating response: {str(e)}", []

def main():
    # Check for API key
    if "COHERE_API_KEY" not in st.secrets:
        st.error("Please set the Cohere API key in your Streamlit secrets")
        st.stop()
    
    # Initialize and run chat interface
    chat_interface = ChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()