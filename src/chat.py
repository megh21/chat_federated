import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere, CohereEmbeddings
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import sqlite3  # For storing session data
from datetime import datetime
import os


class ChatInterface:
    def __init__(self):
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []  # Messages history for the current chat
        if "retriever" not in st.session_state:
            st.session_state.retriever = None
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Initialize Cohere components
        self.embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0",
            cohere_api_key=os.getenv("COHERE_API_KEY"),
        )
        self.chat_model = ChatCohere(
            model="command-r",
            temperature=0.3,  # Lower temperature for more focused responses
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            timeout_seconds=60,  # Increase timeout for longer responses
        )

        # Runnable to manage message history
        self.history_runnable = RunnableWithMessageHistory(memory=st.session_state.messages)
        self.summary = ""

        # Initialize SQLite database for storing sessions
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database for storing chat sessions."""
        with sqlite3.connect("chat_sessions.db") as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT,
                    timestamp TEXT,
                    summary TEXT
                )
            """
            )

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
                (st.session_state.session_id, messages_serialized, timestamp, summary),
            )

    def _load_session(self, session_id: str):
        """Load a chat session from the database."""
        with sqlite3.connect("chat_sessions.db") as conn:
            result = conn.execute(
                "SELECT messages, summary FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if result:
                st.session_state.messages = eval(result[0])  # Deserialize the messages
                st.session_state.session_id = session_id
                st.session_state.summary = result[1]  # Load the summary

    def run(self):
        """Main method to run the chat interface."""
        st.title("Chat with PDFs")
        self._session_selector()

        if not st.session_state.retriever:
            self._setup_database()
        else:
            self._show_chat_interface()

    def _session_selector(self):
        """Allow users to select or restore a previous session."""
        with sqlite3.connect("chat_sessions.db") as conn:
            sessions = conn.execute(
                "SELECT session_id, timestamp, summary FROM sessions"
            ).fetchall()
            session_options = [
                f"{s[0][:9]} (Created on: {s[1][:9]}) - {s[2][:9]}" for s in sessions
            ]

        st.sidebar.subheader("Manage Sessions")
        selected_session = st.sidebar.selectbox(
            "Load Previous Session", ["New Session"] + session_options
        )

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
        """Handle database selection and initialization."""
        st.subheader("Select Database")

        db_dir = Path("databases")
        if not db_dir.exists():
            st.error("No databases directory found. Please create some databases first.")
            return

        databases = [d.name for d in db_dir.iterdir() if d.is_dir()]
        if not databases:
            st.error("No databases found. Please create a database first.")
            return

        selected_db = st.selectbox(
            "Select a database to chat with:", databases, key="database_selector"
        )

        if st.button("Load Database"):
            with st.spinner("Loading database..."):
                try:
                    vector_store = FAISS.load_local(
                        str(db_dir / selected_db),
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    st.session_state.retriever = vector_store.as_retriever(
                        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
                    )
                    st.success("Database loaded successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading database: {str(e)}")

    def _show_chat_interface(self):
        """Display the chat interface and handle messages."""
        st.subheader("Chat Interface")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**\n```{source}```")

        if prompt := st.chat_input("Ask a question about your documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = self._generate_response(prompt)
                    st.write(response)
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**\n```{source}```")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                        }
                    )

            self._save_session()

    def _generate_response(self, question: str) -> Tuple[str, List[str]]:
        """Generate a response using context, history, and source tracking."""
        try:
            docs = st.session_state.retriever.invoke(question)
            chunks_with_ids = [
                f"[{str(uuid.uuid4())[:8]}] {doc.page_content}" for doc in docs
            ]
            context = "\n\n".join(chunks_with_ids)
            chat_history = self.history_runnable.invoke()

            prompt_template = ChatPromptTemplate.from_template("""
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
            """)

            prompt = prompt_template.format(
                context=context, chat_history=chat_history, question=question
            )

            response = self.chat_model.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            sources = [
                doc.page_content
                for doc in docs
                if any(chunk_id in response_text for chunk_id in chunks_with_ids)
            ]

            return response_text.strip(), sources

        except Exception as e:
            return f"Error generating response: {str(e)}", []


def main():
    if os.getenv("OPENAI_API_KEY") is None:
        st.error("Please set the OPENAI API key in your Streamlit secrets")
        st.stop()

    chat_interface = ChatInterface()
    chat_interface.run()


if __name__ == "__main__":
    main()
