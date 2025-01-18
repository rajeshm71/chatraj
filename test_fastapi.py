import requests

# Replace with your actual API base URL
API_BASE_URL = "http://127.0.0.1:8000"

# Initialize session by uploading a file
def test_initialize_chain():
    file_path = r"Data/Case Studies GenAI.docx"  # Replace with your actual file path
    with open(file_path, "rb") as file:
        response = requests.post(
            f"{API_BASE_URL}/initialize/",
            files={"file": file}
        )
    if response.status_code == 200:
        print("Initialization Response:", response.json())
        return response.json().get("session_id")
    else:
        print("Initialization Failed:", response.text)
        return None

# Test streaming endpoint
def test_stream_response(session_id):
    question = "What is the context of the uploaded document?"
    chat_history = []  # Initialize with an empty chat history

    response = requests.post(
        f"{API_BASE_URL}/stream/",
        data={
            "session_id": session_id,
            "question": question,
            "chat_history": chat_history
        },
        stream=True
    )

    if response.status_code == 200:
        print("Streaming response:")
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(line)
    else:
        print(f"Failed to stream response. Status code: {response.status_code}")
        print(f"Response text: {response.text}")

if __name__ == "__main__":
    session_id = test_initialize_chain()
    if session_id:
        test_stream_response(session_id)
