# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from huggingface_hub import login
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_chroma import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import torch
torch.get_default_device = lambda: "cpu"

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
#openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/paper/A_Comprehensive_Exploration_of_Fine-Tuning_WavLM_for_Enhancing_Speech_Emotion_Recognition.pdf"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    #DirectoryLoader(DATA_PATH, glob="*.md", show_progress=True) 
    documents = loader.load()
    
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    #print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    login(os.getenv("HUGGING_FACE_TOKEN")) 

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH, 
    )
    vector_store.add_documents(chunks)
    vector_store.persist()
    # db = Chroma.from_documents(
    #     chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    # )
    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
