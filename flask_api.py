from flask import Flask, request, jsonify, Response
import os
import shutil
from uuid import uuid4
from src.generation_pipeline.generate import RAGChain
import time

# Initialize Flask app
app = Flask(__name__)

# Global storage for RAGChain instances and chat histories
rag_chains = {}
chat_histories = {}

@app.route("/initialize/", methods=["POST"])
def initialize_chain():
    """
    Endpoint to initialize the RAGChain with an uploaded file.

    :return: JSON response with Session ID for the initialized RAGChain.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    session_id = str(uuid4())  # Generate a unique session ID
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file.filename)

    # Save the uploaded file
    file.save(file_path)

    # Initialize a new RAGChain instance
    rag_chain_instance = RAGChain(config_path="config/config.yaml")
    rag_chain_instance.initialize_chain(data_path=data_dir)

    # Store the instance and an empty chat history
    rag_chains[session_id] = rag_chain_instance
    chat_histories[session_id] = []

    return jsonify({"message": f"RAGChain initialized with file: {file.filename}", "session_id": session_id})

@app.route("/stream/", methods=["POST"])
def stream_response():
    """
    Endpoint to stream the response for a given question.

    :return: Streaming response.
    """
    session_id = request.form.get("session_id")
    question = request.form.get("question")
    chat_history = request.form.getlist("chat_history")

    if not session_id or session_id not in rag_chains:
        return jsonify({"error": "Session ID not found. Please initialize the RAGChain first."}), 400

    rag_chain_instance = rag_chains[session_id]

    def generate_stream():
        try:
            # Dynamically create the RAG chain
            rag_chain = rag_chain_instance.run_with_chat_history()

            # Stream the response
            for chunk in rag_chain.stream({"input": question, "chat_history": chat_history}):
                chunk_content = chunk.get('answer', '')
                print(chunk_content)
                yield chunk_content + "\n"
                time.sleep(0.1)  # Simulate delay for streaming
        except Exception as e:
            yield f"Error: {str(e)}\n"

    return Response(generate_stream(), content_type="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=8001)
