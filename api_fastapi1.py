from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
import time

app = FastAPI()

# Generator function to stream data


@app.get("/stream")
def stream_response():


    def data_stream():
        for i in range(10):
            yield f"data: Message {i}\n\n"  # Streaming data line by line
            time.sleep(1)  # Simulate delay for streaming
    return StreamingResponse(data_stream(), media_type="text/plain")
