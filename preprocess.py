# preprocess.py

# ─── Override sqlite3 with pysqlite3 (ensures sqlite3 ≥ 3.35.0) ───────────────────
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ─── Standard imports ─────────────────────────────────────────────────────────────
import os
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma  # correct Chroma import
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_vectorstore(pdf_path: str, persist_dir: str):
    # ─── Load API key ─────────────────────────────────────────────────────────────
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in .env")

    # ─── Read and concatenate PDF text ────────────────────────────────────────────
    reader = PyPDF2.PdfReader(pdf_path)
    text = "\n\n".join(page.extract_text() for page in reader.pages)

    # ─── Split into chunks ────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = splitter.split_text(text)

    # ─── Embed & build Chroma vectorstore (auto‑persists to `persist_dir`) ───────
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=google_api_key
    )
    _ = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="gigfinder"
    )
    # No need to call `.persist()`—Chroma writes to disk on creation if you pass `persist_directory` :contentReference[oaicite:1]{index=1}
    print(f"✅ Vectorstore built and persisted to '{persist_dir}'")

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
