from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

def create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(
        documents
    )

    for chunk in chunks:
        chunk.metadata["source"] = pdf_path

    embedding_model = MistralAIEmbeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="vectorstore"
    )

    return len(chunks)

