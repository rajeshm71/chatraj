import streamlit as st
import requests
import datetime

# Base URL for FastAPI endpoints
API_BASE_URL = "http://127.0.0.1:8000"

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Function to initialize the chain
def initialize_chain(file):
    with st.spinner("Uploading file..."):
        response = requests.post(f"{API_BASE_URL}/initialize/", files={"file": file})
    if response.status_code == 200:
        data = response.json()
        st.session_state["session_id"] = data.get("session_id")
        st.success(data.get("message", "Uploaded file successfully!"))
    else:
        st.error(f"Error initializing chain: {response.text}")

# Function to generate an answer
def generate_answer(question):
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.error("Chain is not initialized. Please upload a file first.")
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    response = requests.post(
        f"{API_BASE_URL}/stream/",
        data={"session_id": session_id, "question": question, "chat_history": st.session_state.chat_history},
        stream=True,
    )

    if response.status_code == 200:
        for chunk in response.iter_lines(decode_unicode=True):
            chunk_content = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            print(chunk_content)
            yield chunk_content

    else:
        st.error(f"Error querying the chain: {response.text}")
        return
# Streamlit Page Configuration
st.set_page_config(page_title="Conversational RAGChain App", page_icon="💬", layout="wide")

# Sidebar: Upload Section
st.sidebar.header("📂 Upload Context File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file to provide context (e.g., .txt, .pdf, .docx):", type=["txt", "pdf", "docx"]
)
if uploaded_file and st.sidebar.button("Upload"):
    initialize_chain(uploaded_file)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Rajesh")

# Main App Layout
st.title("💬 Conversational RAGChain App")

# tab1, tab2 = st.tabs(["Chat", "Instructions"])

# Chat Tab
st.header("Chat with your document...")
if st.session_state.get("session_id"):
    for message in st.session_state.chat_history:
        #role_style = "color: blue;" if message["role"] == "assistant" else "color: green;"
        role_style = "color: black;"
        avatar = "🧑" if  message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(f"<p style='{role_style}'>{message['content']}</p>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question:")
    if user_input:
        with st.chat_message("user",avatar="🧑"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="🤖"):
            full_response = st.write_stream( generate_answer(user_input))
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Instructions Tab
# with tab2:
#     st.markdown("### How to Use the App")
#     st.markdown("""
#     1. Upload a context file from the sidebar to initialize the chain.
#     2. Once initialized, start asking questions in the chat section.
#     3. Responses will appear in real-time.
#     4. Use the sidebar to reset or re-upload context as needed.
#     """)

# Footer
# st.markdown("<footer style='text-align: center; margin-top: 50px;'>Built using Streamlit 🚀</footer>", unsafe_allow_html=True)
