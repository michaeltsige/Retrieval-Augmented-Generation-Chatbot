# preprocess.py

import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

def build_vectorstore(pdf_path: str, persist_dir: str):
    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in .env")

    # Read the PDF
    reader = PyPDF2.PdfReader(pdf_path)
    text = "\n\n".join(page.extract_text() for page in reader.pages)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(text)

    # Embed & build Chroma vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=google_api_key
    )
    vectordb = Chroma.from_texts(
        docs,
        embeddings,
        persist_directory=persist_dir,
        collection_name="gigfinder"
    )
    vectordb.persist()
    print(f"Vectorstore built and persisted to '{persist_dir}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default=os.path.join("data", "data.pdf"),
        help="Path to your GigFinder PDF (default: data/data.pdf)"
    )
    parser.add_argument(
        "--out",
        default="vectorstore",
        help="Where to persist Chroma files (default: vectorstore/)"
    )
    args = parser.parse_args()

    build_vectorstore(args.pdf, args.out)
