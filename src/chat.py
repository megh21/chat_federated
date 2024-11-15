import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere, CohereEmbeddings
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import sqlite3  # For storing session data
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import os

class ChatInterface:
    def __init__(self):
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Initialize Cohere components
        self.embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0",
            cohere_api_key= os.getenv("COHERE_API_KEY")
        )
        self.chat_model = ChatCohere(
            model="command-r",
            temperature=0.3,  # Lower temperature for more focused responses
            cohere_api_key= os.getenv("COHERE_API_KEY"),
            timeout_seconds=60  # Increase timeout for longer responses

        )
        self.persistent_chat_history = ChatMessageHistory()
        self.summary = ""

        # Initialize SQLite database for storing sessions if it doesn't exist
        self._init_database()


    def _init_database(self):
        """Initialize the SQLite database for storing chat sessions."""
        with sqlite3.connect("chat_sessions.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT,
                    timestamp TEXT,
                    summary TEXT
                )
            """)

    def _save_session(self):
        """Save the current chat session to the database."""
        with sqlite3.connect("chat_sessions.db") as conn:
            messages_serialized = str(st.session_state.messages)  # Serialize the messages
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = self.summary
            conn.execute(
                """
                INSERT INTO sessions (session_id, messages, timestamp, summary)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET messages = excluded.messages, timestamp = excluded.timestamp, summary = excluded.summary
                """,
                (st.session_state.session_id, messages_serialized, timestamp, summary)
            )

    def _load_session(self, session_id: str):
        """Load a chat session from the database."""
        with sqlite3.connect("chat_sessions.db") as conn:
            result = conn.execute(
                "SELECT messages, summary FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if result:
                st.session_state.messages = eval(result[0])  # Deserialize the messages
                st.session_state.session_id = session_id
                st.session_state.summary = result[1]  # Load the summary

    def run(self):
        """Main method to run the chat interface"""
        st.title("Chat with PDFs")
        # Load previous session if available
        self._session_selector()

        # First, handle database selection
        if not st.session_state.retriever:
            self._setup_database()
        else:
            self._show_chat_interface()

    def _session_selector(self):
        """Allow users to select or restore a previous session."""
        with sqlite3.connect("chat_sessions.db") as conn:
            sessions = conn.execute("SELECT session_id, timestamp, summary FROM sessions").fetchall()
            session_options = [f"{s[0][:9]} (Created on: {s[1][:9]}) - {s[2][:9]}" for s in sessions]
    
        st.sidebar.subheader("Manage Sessions")
        selected_session = st.sidebar.selectbox("Load Previous Session", ["New Session"] + session_options)
    
        if selected_session != "New Session":
            session_id = selected_session.split(" ")[0]
            if st.sidebar.button("Load Session"):
                self._load_session(session_id)
                st.experimental_rerun()
        elif st.sidebar.button("Start New Session"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.experimental_rerun()

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
        
                # Check and remove the last error message if present, only when a new input is provided
        if st.session_state.messages and st.session_state.messages[-1]["content"].startswith("Error generating response"):
            # Remove the last assistant error message and its associated user query
            st.session_state.messages.pop()  # Remove assistant error message
            st.session_state.messages.pop()  # Remove user query

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
                    if not response.startswith("Error"):
                        # Add assistant's response to chat
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
                    else:
                        # Handle error messages in a graceful way
                        st.write(response)  # Display the error response but avoid saving it

            # Save the session after each message
            self._save_session()

    def summarize_messages(self) -> str:
        """Summarize the chat history for context preparation."""
        try:
            if len(st.session_state.messages) == 0:
                return ""

            # Combine all messages into a single text block
            chat_history_text = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
            )

            # Create summarization prompt
            summarization_prompt = f"""
            Summarize the following chat history into a concise summary, retaining key details:

            {chat_history_text}
            """
            # Invoke the chat model for summarization
            response = self.chat_model.invoke(summarization_prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            print(f"Error summarizing messages: {e}")
            return ""


    def trim_chat_history(self, max_tokens: int = 500) -> str:
        """Trim the chat history to fit within the token limit."""
        try:
            # Extract chat messages as text
            chat_history_text = [
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages
            ]

            # Use only the last `max_tokens` worth of tokens
            current_tokens = 0
            trimmed_history = []
            for message in reversed(chat_history_text):
                message_tokens = len(message.split())  # Approximation using words
                if current_tokens + message_tokens > max_tokens:
                    break
                trimmed_history.insert(0, message)
                current_tokens += message_tokens

            return "\n".join(trimmed_history)

        except Exception as e:
            print(f"Error trimming chat history: {e}")
            return ""



    def _generate_response(self, question: str) -> Tuple[str, List[str]]:
        """Generate a response using RAG with source tracking and prepared context."""
        if st.session_state.messages and st.session_state.messages[-1]["content"].startswith("Error generating response"):
            st.session_state.messages.pop()  # Remove last assistant error message
            st.session_state.messages.pop()  # Remove corresponding user message

        try:
            # Log the input question
            print(f"Question: {question}")

            # Get relevant documents
            docs = st.session_state.retriever.invoke(question)
            print(f"Retrieved {len(docs)} documents for the query.")

            # Create context with unique IDs
            chunks_with_ids = []
            source_map = {}
            
            for doc in docs:
                chunk_id = str(uuid.uuid4())[:8]
                chunks_with_ids.append(f"[{chunk_id}] {doc.page_content}")
                source_map[chunk_id] = doc.page_content
            
            context = "\n\n".join(chunks_with_ids)

            # Summarize or trim chat history for context preparation
            if len(st.session_state.messages) > 5:
                chat_history = self.summarize_messages()
                self.summary =chat_history
                print("Chat history was summarized.")
            else:
                chat_history = self.trim_chat_history(max_tokens=500)
                print("Chat history was trimmed.")

            # Prepare the prompt with chat history and document context
            prompt = f"""
            You are a helpful research assistant. Answer the following question based on the provided context
            and chat history. Use information ONLY from the provided context. If you're not sure, say so.

            Context:
            {context}

            Chat History:
            {chat_history}

            Question: {question}

            Instructions:
            1. Answer the question using ONLY the provided context
            2. Reference source IDs [xxxx] when using specific information
            3. If you can't answer from the context, say so
            4. Be concise but thorough
            5. Use analytical and scientific language
            """

            print(f"Prompt:\n{prompt}")
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
            # Log the error
            error_message = f"Error generating response: {str(e)}"
            print(error_message)

            # Store the error in session state for cleanup on next query
            st.session_state.error_occurred = True
            st.session_state.last_error = {"role": "assistant", "content": error_message}
            return error_message, []

def main():
    # Check for API key
    if os.getenv("COHERE_API_KEY") is None:
        st.error("Please set the Cohere API key in your Streamlit secrets")
        st.stop()
    
    # Initialize and run chat interface
    chat_interface = ChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()