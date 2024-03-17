import getpass
import os
import dotenv
import streamlit as st
from langchain_community.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import CohereEmbeddings
import warnings

warnings.filterwarnings("ignore")

def load_environment():
    """
    Load the environment variables.
    If the COHERE_API_KEY is not set, prompt the user to enter it.
    """
    dotenv.load_dotenv()
    if "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = getpass.getpass()

def initialize_chatbot():
    """
    Initialize the chatbot and the retriever.
    """
    chat = ChatCohere(model="command", max_tokens=256, temperature=0.75)
    embedding = CohereEmbeddings(model="embed-english-light-v3.0")
    db = FAISS.load_local("store/vectorstore", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    return chat, retriever

def format_docs(docs):
    """
    Format the documents for display.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_prompt_template():
    """
    Create the prompt template for the chatbot.
    """
    template = """
    You are a Question Answering bot. You answer in complete sentences and step-by-step whenever necessary.
    You always provide references from the Research papers and required docs which are Literature review material about the thesis on Federated Learning with page number and paragraph start sentence in 
    double quotation marks. 
    Answer the question based only on the following context, which can include information about Data:
    {context}
    Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

def create_rag_chain(chat, retriever, prompt_template):
    """
    Create the RAG chain for the chatbot.
    """
    return (
        {"context": retriever | format_docs, "question": chat}
        | prompt_template
        | chat
    )

def create_ui(rag_chain):
    """
    Create the user interface for the chatbot.
    """
    st.title("Federated Learning Literature Chatbot")
    st.image("logos/logo_insights.png", use_column_width=True, width=200)

    st.sidebar.image("logos/shelf.png",  width=200)
    st.sidebar.markdown("[Go to the report](report_link_here)")

    st.sidebar.markdown("## Navigation")
    if st.sidebar.button("Go to File Converter"):
        st.rerun()

    st.markdown("### Ask a Question")
    user_question = st.text_input("Type your question here:")
    if user_question:
        response = rag_chain.invoke(user_question)
        st.markdown("**Bot's Response:**")
        st.write(response)

def chatbot_page():
    """
    Main function for the chatbot page.
    """
    load_environment()
    chat, retriever = initialize_chatbot()
    prompt_template = create_prompt_template()
    rag_chain = create_rag_chain(chat, retriever, prompt_template)
    create_ui(rag_chain)

if __name__ == "__main__":
    chatbot_page()