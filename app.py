# app.py

import os
import sys
import pysqlite3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Workaround for SQLite driver on some systems
sys.modules["sqlite3"] = pysqlite3

# Load env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# FastAPI app
app = FastAPI()

# CORS (allow frontend apps to connect, update as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vectorstore (once, on startup)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
retriever = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
    collection_name="gigfinder"
).as_retriever(search_kwargs={"k": 5})

# Prompt template
prompt = PromptTemplate(
    template="""
You are the Gig Marketplace chatbot. Respond clearly, concisely, and conversationally.

User's Question: {question}

Context: {context}

If context isn't sufficient, politely say so and offer help.
""",
    input_variables=["context", "question"]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, api_key=api_key)
qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            question = data.get("question")

            if not question:
                await websocket.send_json({"error": "Missing 'question'"})
                continue

            # RAG pipeline
            docs = retriever.get_relevant_documents(question)
            result = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)

            await websocket.send_json({"answer": result["output_text"]})
    except WebSocketDisconnect:
        print("Client disconnected")
