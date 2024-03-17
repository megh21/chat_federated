import os
import dotenv
import getpass
import streamlit as st
from store_files import file_converter_page
from ChatbotUI import chatbot_page

def load_api_key():
    """
    Load the API key from the environment variables.
    If it's not set, prompt the user to enter it.
    """
    dotenv.load_dotenv()
    if "COHERE_API_KEY" not in os.environ:
        os.environ["COHERE_API_KEY"] = getpass.getpass()

def display_page(page_function):
    """
    Display the selected page.
    """
    page_function()

def main():
    """
    Main function for the Streamlit app.
    """
    load_api_key()

    st.title("Multi-Page App")

    # Define the page options and their corresponding functions
    page_options = {
        "File Converter": file_converter_page,
        "Chatbot": chatbot_page
    }

    # Let the user select a page
    selected_page = st.sidebar.selectbox("Select Page", list(page_options.keys()))

    # Display the selected page
    display_page(page_options[selected_page])

if __name__ == "__main__":
    main()