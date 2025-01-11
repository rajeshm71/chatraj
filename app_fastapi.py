from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import os
import shutil
from typing import AsyncGenerator
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from src.generation_pipeline.generate import RAGChain
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Global storage for RAGChain instances and chat histories
rag_chains = {}
chat_histories = {}

@app.post("/initialize/")
async def initialize_chain(file: UploadFile):
    """
    Endpoint to initialize the RAGChain with an uploaded file.

    :param file: The uploaded file containing context.
    :return: Session ID for the initialized RAGChain.
    """
    session_id = str(uuid4())  # Generate a unique session ID
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize a new RAGChain instance
    rag_chain_instance = RAGChain(config_path="config/config.yaml")
    rag_chain_instance.initialize_chain(data_path=data_dir)

    # Store the instance and an empty chat history
    rag_chains[session_id] = rag_chain_instance
    chat_histories[session_id] = []

    return {"message": f"RAGChain initialized with file: {file.filename}", "session_id": session_id}


@app.post("/stream/")
async def stream_response(session_id: str = Form(...), question: str = Form(...)) -> StreamingResponse:
    """
    Endpoint to stream the response for a given question.

    :param session_id: The session ID for the user's RAGChain.
    :param question: The question to query the RAGChain.
    :return: Streaming response.
    """
    if session_id not in rag_chains:
        return {"error": "Session ID not found. Please initialize the RAGChain first."}

    rag_chain_instance = rag_chains[session_id]

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Retrieve the chat history for this session
            chat_history = chat_histories[session_id]

            # Dynamically create the RAG chain
            rag_chain = rag_chain_instance.run_with_chat_history()

            # Stream the response
            for chunk in rag_chain.stream({"input": question, "chat_history": chat_history}):
                chunk_content = chunk.get('answer', '')
                yield chunk_content

                # Dynamically update chat history
                if chunk_content.strip():
                    chat_history.append(HumanMessage(content=question))
                    chat_history.append(AIMessage(content=chunk_content))
                # await asyncio.sleep(0.5)
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_stream(), media_type="text/plain")
