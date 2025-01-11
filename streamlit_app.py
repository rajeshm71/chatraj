import streamlit as st
import requests
import asyncio

# Base URL for FastAPI endpoints
API_BASE_URL = "http://127.0.0.1:8000"

# Session state to store session ID and chat history
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "initialized" not in st.session_state:
    st.session_state["initialized"] = False


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
        st.session_state["initialized"] = True
        st.success(data.get("message", "Chain initialized successfully!"))
    else:
        st.error(f"Error initializing chain: {response.text}")


def generate_answer(question):
    """
    Query the RAGChain API and stream the response.
    """
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.error("Chain is not initialized. Please upload a file first.")
        return

    # Make a POST request to the streaming endpoint
    response = requests.post(
        f"{API_BASE_URL}/stream/",
        data={"session_id": session_id, "question": question},
        stream=True
    )

    if response.status_code == 200:
        full_response = ""
        st.info("Generating answer...")
        for chunk in response.iter_lines():
            chunk_content = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            print("chunk_content")
            st.write(chunk_content)  # Display chunk
            full_response += chunk_content
        st.success("Answer generation complete!")
        return full_response
    else:
        st.error(f"Error querying the chain: {response.text}")


# Streamlit App Layout
st.title("RAGChain Question-Answering System")

# Upload File Section
st.header("1. Upload Context File")
uploaded_file = st.file_uploader("Upload a file to provide context (e.g., .txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if uploaded_file and not st.session_state["initialized"]:
    if st.button("Initialize Chain"):
        initialize_chain(uploaded_file)

# Question Section
if st.session_state["initialized"]:
    st.header("2. Ask a Question")
    question = st.text_input("Enter your question:")

    if question and st.button("Generate Answer"):
        generate_answer(question)
