# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_cohere import ChatCohere, CohereRagRetriever
# from langchain.prompts import ChatPromptTemplate
# import json
# import os
# from langchain_cohere import CohereEmbeddings
# from langchain.schema import ChatMessage
# from pprint import pprint

# def start_existing_chat_session(session_name):
#     print(f"Starting chat session: {session_name}")
#     with open(f"sessions/{session_name}", "r") as session_file:
#         session_data = json.load(session_file)
#     display_chat_interface(session_data=session_data)

# def start_chat_interface():
#     st.subheader("Chat Interface")
#     print("Starting chat interface")

#     # Ensure the 'databases' directory exists
#     if not os.path.exists("databases"):
#         st.error("The 'databases' directory does not exist.")
#         return

#     # Get available databases
#     database_options = [f for f in os.listdir("databases") if os.path.isdir(os.path.join("databases", f))]
#     if not database_options:
#         st.error("No databases found in the 'databases' directory.")
#         return

#     # User selects a database
#     selected_database = st.selectbox("Select a database to use", database_options)
#     print("Selected database:", selected_database)

#     # Define callback to set session state on button click
#     def start_session_callback():
#         if selected_database:
#             print("Loading database:", selected_database)
#             st.write(f"Loading database: {selected_database}")

#             # Load the database and set session state
#             retriever = FAISS.load_local(
#                 f"databases/{selected_database}",
#                 embeddings=CohereEmbeddings(model="embed-english-light-v3.0"),
#                 allow_dangerous_deserialization=True
#             ).as_retriever()

#             # Update session states
#             st.session_state['retriever'] = retriever
#             st.session_state['session_started'] = True
#             st.session_state['chat_interface_active'] = False
#             st.write("Session started with retriever loaded.")
#         else:
#             st.error("Please select a database to start the chat session.")

#     # Button to start chat session with callback
#     st.button("Start Chat Session", on_click=start_session_callback)
#     print("End of start_chat_interface function")


# def save_chat_session(session_data, session_name):
#     print(f"Saving chat session: {session_name}")
#     with open(f"sessions/{session_name}", "w") as session_file:
#         json.dump(session_data, session_file)
#     st.success("Chat session saved successfully.")


# def display_chat_interface(retriever=None, session_data=None):
#     print("Displaying chat interface")
#     st.subheader("Active Chat Session")

#     if retriever is None:
#         st.error("Retriever is not available. Please start a new session.")
#         return

#     # Display chat input and history
#     question = st.text_input("Enter your question", key="user_question")
#     if st.button("Send"):
#         handle_user_query(retriever)
    
#     # Show session data if it exists
#     if session_data:
#         for message in session_data.get("messages", []):
#             st.write(message)
#     #save to session data
#     if not session_data:
#         session_data = {"messages": []}
#     if question:
#         session_data["messages"].append(ChatMessage(question, "user"))
#     # Save the chat session
#     if st.button("Save Chat Session",key="save_chat_session_btn"):
#         save_chat_session(session_data, session_name="chat_session.json")


# # Function to handle queries with context length limit for each document
# def query_with_context_limit(documents, token_limit=1000):
#     # Calculate the token limit for each document
#     doc_token_limit = token_limit // len(documents)
    
#     for doc in documents:
#         tokens = doc.page_content.split()
#         if len(tokens) > doc_token_limit:
#             truncated_content = " ".join(tokens[:doc_token_limit])
#         else:
#             truncated_content = doc.page_content
#         # Update the document's page_content in place
#         doc.page_content = truncated_content
    
#     return documents



# def handle_user_query(retriever):
#     # try:
#     print("Handling user query")
#     # Wherever you set the prompt template in your main code, it should now look like this:
#     prompt_template = create_prompt_template()

    
#     # Retrieve the question from session state
#     question = st.session_state.get("user_question")
#     if not question:
#         st.warning("Please enter a question.")
#         return

#     # Step 1: Retrieve relevant documents
#     # print("Retrieving documents for the question:", question)


#     relevant_docs = retriever.invoke(question)  # Updated to use invoke method
#     if not relevant_docs:
#         st.warning("No relevant documents found.")
#         return
#     print("Retrieved documents:", len(relevant_docs))
#     print('relevant_docs: \n ')
#     # print( relevant_docs)

#     relevant_docs=query_with_context_limit(relevant_docs)
#     print('relevant_docs after query_with_context_limit: \n ')
#     # print( relevant_docs)
#     # Step 2: Generate a response using the language model
#     print("Generating response from retrieved documents")
    
#     # Create a prompt manually from documents and question
#     prompt_text = prompt_template.format(
#     context="\n".join([doc.page_content for doc in relevant_docs]),  # Add document contents as context
#     question=question
#     )
#     # print("Prompt text:", prompt_text)
#     # Initialize and use ChatCohere with the constructed prompt
    
#     # response = chat_model.generate(prompt_text)
#     chat_model = ChatCohere(model="command-r", cohere_api_key=os.getenv("COHERE_API_KEY"))
#     rag = CohereRagRetriever(llm=chat_model)
#     response = rag.invoke(prompt_text,documents=relevant_docs)

#     # Display the response
#     st.write("Response:", response)
#     print(type(response))
#     print()
#     pprint(response)
#     # except Exception as e:
#     #     print(f"Error in handle_user_query: {e}")



#********1st try************

# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever
# from langchain.prompts import ChatPromptTemplate
# from pathlib import Path
# import os

# class ChatInterface:
#     def __init__(self):
#         # Initialize session state
#         if 'messages' not in st.session_state:
#             st.session_state.messages = []
#         if 'retriever' not in st.session_state:
#             st.session_state.retriever = None
        
#         # Initialize Cohere components
#         self.embeddings = CohereEmbeddings(
#             model="embed-english-light-v3.0",
#             cohere_api_key=st.secrets["COHERE_API_KEY"]
#         )
#         self.chat_model = ChatCohere(
#             model="command-r",
#             cohere_api_key=st.secrets["COHERE_API_KEY"]
#         )

#     def run(self):
#         """Main method to run the chat interface"""
#         st.title("Chat with PDFs")
        
#         # First, handle database selection
#         if not st.session_state.retriever:
#             self._setup_database()
#         else:
#             self._show_chat_interface()

#     def _setup_database(self):
#         """Handle database selection and initialization"""
#         st.subheader("Select Database")
        
#         # Check for databases directory
#         db_dir = Path("databases")
#         if not db_dir.exists():
#             st.error("No databases directory found. Please create some databases first.")
#             return
        
#         # Get available databases
#         databases = [d.name for d in db_dir.iterdir() if d.is_dir()]
#         if not databases:
#             st.error("No databases found. Please create a database first.")
#             return
        
#         # Database selection
#         selected_db = st.selectbox(
#             "Select a database to chat with:",
#             databases,
#             key="database_selector"
#         )
        
#         # Load database button
#         if st.button("Load Database"):
#             with st.spinner("Loading database..."):
#                 try:
#                     vector_store = FAISS.load_local(
#                         str(db_dir / selected_db),
#                         embeddings=self.embeddings,
#                         allow_dangerous_deserialization=True
#                     )
#                     st.session_state.retriever = vector_store.as_retriever(
#                         search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
#                     )
#                     st.success("Database loaded successfully!")
#                     st.experimental_rerun()
#                 except Exception as e:
#                     st.error(f"Error loading database: {str(e)}")

#     def _show_chat_interface(self):
#         """Display the chat interface and handle messages"""
#         st.subheader("Chat Interface")
        
#         # Display chat history
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])
        
#         # Chat input
#         if prompt := st.chat_input("Ask a question about your documents"):
#             # Add user message to chat
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.write(prompt)
            
#             # Generate response
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     response = self._generate_response(prompt)
#                     st.write(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})

#     def _generate_response(self, question: str) -> str:
#         """Generate a response using RAG"""
#         try:
#             # Get relevant documents
#             documents = st.session_state.retriever.invoke(question)
            
#             # Prepare RAG retriever
#             rag = CohereRagRetriever(llm=self.chat_model)
            
#             # Generate response
#             prompt = self._create_prompt(question, documents)
#             response = rag.invoke(prompt, documents=documents)
            
#             return response.response if hasattr(response, 'response') else str(response)
            
#         except Exception as e:
#             return f"Error generating response: {str(e)}"

#     def _create_prompt(self, question: str, documents: list) -> str:
#         """Create a prompt for the language model"""
#         context = "\n\n".join(doc.page_content for doc in documents)
        
#         template = """
#         Based on the following context, please provide a clear and accurate answer to the question.
#         If the answer cannot be derived from the context, please say so.

#         Context:
#         {context}

#         Question: {question}

#         Answer in a clear and concise manner, citing specific parts of the context when relevant.
#         """

#         #     template = """
# #     You are an advanced Research Assistant, designed for answering complex scientific questions across various domains and languages.
# #     Your responses are clear, comprehensive, and cite references meticulously to ensure academic integrity. 
# #     Follow these guidelines in your answers:

# #     1. Answer in complete sentences and provide explanations step-by-step wherever needed.
# #     2. Use references from relevant research papers in the specified database, indicating the source title, author(s), publication year, page number, and section or paragraph, as available.
# #     3. Ensure that references are specific, citing paper details in double quotation marks with relevant identifiers.
# #     4. Summarize only the necessary information related to the query, filtering out any unrelated data.
# #     5. For multilingual sources, preserve the context by accurately summarizing or quoting from the original language where necessary, but provide responses in the language of the question or in English if unspecified.

# #     Context (literature and data available):
# #     {context}

# #     Question:
# #     {question}

# #     Expected Output:
# #     - Direct answer to the question, drawn only from the given context.
# #     - Reference(s) to specific paper(s) that support your answer, formatted as:
# #       - "Title of the Paper" by Author(s), Published Year, Page X, Paragraph starting with: "..."
# #     - Clear, concise explanation based on synthesized information if multiple papers contribute to the answer.
# #     """
        
#         return template.format(context=context, question=question)
    


# def main():
#     # Check for API key
#     if "COHERE_API_KEY" not in st.secrets:
#         st.error("Please set the Cohere API key in your Streamlit secrets")
#         st.stop()
    
#     # Initialize and run chat interface
#     chat_interface = ChatInterface()
#     chat_interface.run()

# if __name__ == "__main__":
#     main()


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
            temperature=0.1,  # Lower temperature for more focused responses
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