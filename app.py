import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# load env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

# Fastapi setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generic responses
GENERIC_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! What can I do for you?",
    "hey": "Hey! How can I assist you?",
    "how are you": "I'm here and ready to help! How can I assist?",
    "who are you": "I'm your Gig Marketplace assistant, here to help you navigate gigs and questions.",
    "what is your name": "I'm the GigFinder chatbot!",
    "thanks": "You're welcome!",
    "thank you": "Happy to help!",
}

def detect_generic_response(text: str):
    cleaned = text.lower().strip()

    # Exact match
    if cleaned in GENERIC_RESPONSES:
        return GENERIC_RESPONSES[cleaned]

    # Partial match
    for key in GENERIC_RESPONSES:
        if key in cleaned:
            return GENERIC_RESPONSES[key]

    return None

# retreiver
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=api_key
)
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings,
    collection_name="gigfinder"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = PromptTemplate(
    template="""
You are the Gig Marketplace chatbot. Respond clearly, concisely, and conversationally.

User's Question: {input}

Context: {context}

If context isn't sufficient, politely say so and offer help.
""",
    input_variables=["context", "input"]
)

# Init LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3,
    api_key=api_key
)

# RAG chain
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

# Web socket
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("question")

            if not user_input:
                await websocket.send_json({"error": "Missing 'question'"})
                continue

            generic = detect_generic_response(user_input)
            if generic:
                await websocket.send_json({"answer": generic})
                continue

            try:
                result = rag_chain.invoke({"input": user_input})
                await websocket.send_json({"answer": result["answer"]})

            except Exception as e:
                error_msg = str(e).lower()

                if (
                    "quota" in error_msg
                    or "exceed" in error_msg
                    or "429" in error_msg
                    or "resource exhausted" in error_msg
                ):
                    await websocket.send_json({
                        "answer": "API limit exceeded. Please try again later."
                    })
                else:
                    await websocket.send_json({
                        "answer": f"Unexpected error: {str(e)}"
                    })

    except WebSocketDisconnect:
        print("Client disconnected")
