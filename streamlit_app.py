import streamlit as st
import requests
from src.generation_pipeline.generate import RAGChain
import json

# Base URL for FastAPI endpoints
API_BASE_URL = "http://127.0.0.1:8000"

# Session state to store session ID and chat history
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to initialize the chain
def initialize_chain(file):
    """
    Initialize the RAGChain with an uploaded file.
    """
    with st.spinner("Initializing chain with context file..."):
        response = requests.post(
            f"{API_BASE_URL}/initialize/",
            files={"file": file}
        )
    if response.status_code == 200:
        data = response.json()
        st.session_state["session_id"] = data.get("session_id")
        st.success(data.get("message", "Chain initialized successfully!"))
    else:
        st.error(f"Error initializing chain: {response.text}")

# Function to generate an answer
def generate_answer(question):
    """
    Query the RAGChain API and stream the response in real-time.
    """
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.error("Chain is not initialized. Please upload a file first.")
        return

    # Add the user's question to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})

    # # Define a generator to stream the response
    # rag_chain_instance = RAGChain(config_path="config/config.yaml")
    
    response = requests.post(
        f"{API_BASE_URL}/stream/",
        data=({"session_id": session_id, "question": question, "chat_history": st.session_state.chat_history}),
        stream=True
    )
    
    if response.status_code == 200:
        for chunk in response.iter_lines(decode_unicode=True):
            chunk_content = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            print(chunk_content)
            yield chunk_content

    else:
        st.error(f"Error querying the chain: {response.text}")
        return
        # rag_chain = rag_chain_instance.run_with_chat_history()
        # return rag_chain.stream({"input": question, "chat_history": st.session_state.chat_history})
        
        
    # Stream the response using st.write_stream
    # full_response = st.write_stream(response_generator())

    #Add the assistant's response to chat history
    # st.session_state.chat_history.append({"role": "assistant", "content": full_response})
   


# Streamlit App Layout
st.set_page_config(page_title="Conversational RAGChain App", page_icon="💬", layout="centered")

st.title("💬 Conversational RAGChain App")

# File Upload Section
st.sidebar.header("Upload Context File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file to provide context (e.g., .txt, .pdf, .docx):",
    type=["txt", "pdf", "docx"]
)
if uploaded_file:
    if st.sidebar.button("Initialize Chain"):
        initialize_chain(uploaded_file)

# Chat Section
st.header("Chat with the RAGChain")
if st.session_state.get("session_id"):
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_input = st.chat_input("Ask a question:")
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # response = generate_answer(user_input)
            #     st.chat_message("assistant").markdown(response)
        with st.chat_message("assistant"):
            full_response = st.write_stream(generate_answer(user_input))
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Display Chat History
    # for message in st.session_state.chat_history:
    #     if message["role"] == "user":
    #         st.chat_message("user").markdown(message["content"])
    #     elif message["role"] == "assistant":
    #         st.chat_message("assistant").markdown(message["content"])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Made with ❤️ by Rajesh"
)
