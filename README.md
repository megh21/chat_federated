# Federated Learning Thesis Chatbot

This repository contains the code for a federated learning thesis chatbot implemented using the RAG (Retrieval-Augmented Generation) model.

## Overview

The goal of this project is to develop a chatbot that utilizes LLm techniques to effectively chat with thesis literature.

The RAG model is used as the underlying architecture, which combines retrieval-based and generation-based approaches to generate responses.

## Technologies Used

- FAISS 
- Langchain
- Cohere API
- Sentence Transformers
- Hugging Face


**The demo is Live at [koyeb](https://chatfederated-megh.koyeb.app/)**
(hopefully the link is working, can't guarantee if it will still work. if it does not work, please try running cloning & running offline 

## Features
- RAG model: The RAG model is used to generate responses by combining retrieval-based and generation-based techniques.
- Dataset: The chatbot is trained on a dataset of conversational data specific to the thesis topic.

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/megh21/chat-federated.git
    ```
2. Create a conda environment using the provided `env.yaml` file:

    ```shell
    conda env create -f env.yaml
    ```
3. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

4. Add API keys to the dotenv file:
    - Create a `.env` file in the root directory of the project.
    - Add the following lines to the `.env` file, replacing `<API_KEY>` with your actual API key:

        ```plaintext
        COHERE_API_KEY=<API_KEY>
        ```

    - Make sure to keep the `.env` file secure and do not commit it to version control.

5. Continue with the usage instructions below.


## Usage

1. Preprocess the dataset (optional):

    ```shell
    python preprocess.py --dataset <path_to_dataset>
    ```

2. load the vector files:

    ```shell
    python store_files.py
    ```

3. Start the chatbot:

    ```shell
    streamlit run main.py
    ```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the RAG model implementation.
- koyeb free tier hosting for demo.
- cohere trial API key for demo.
- My supervisor for guidance and support throughout the thesis.
