import requests

# Replace with the Flask server's URL
BASE_URL = "http://127.0.0.1:8001"

def test_initialize_chain():
    """
    Test the `/initialize/` endpoint by uploading a file and returning the session ID.
    """
    file_path = r"Data/Case Studies GenAI.docx"  # Replace with your actual file path
    with open(file_path, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/initialize/",
            files={"file": file}
        )
    if response.status_code == 200:
        print("Initialization Response:", response.json())
        return response.json().get("session_id")
    else:
        print("Initialization Failed:", response.text)
        return None

def test_stream_response(session_id):
    """
    Test the `/stream/` endpoint by sending a question and printing streamed responses.
    """
    question = "What is the context of the uploaded document?"
    chat_history = []  # Example: empty chat history to start

    response = requests.post(
        f"{BASE_URL}/stream/",
        data={
            "session_id": session_id,
            "question": question,
            "chat_history": chat_history
        },
        stream=True  # Enable streaming
    )

    if response.status_code == 200:
        print("Streaming Response:")
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Filter out keep-alive lines
                    print(line)
        except Exception as e:
            print(f"Error while streaming: {str(e)}")
    else:
        print(f"Failed to stream response. Status code: {response.status_code}")
        print(f"Response text: {response.text}")

if __name__ == "__main__":
    # Test initialization
    session_id = test_initialize_chain()
    if session_id:
        # Test streaming if initialization is successful
        test_stream_response(session_id)
