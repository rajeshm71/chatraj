import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
print(OPENAI_API_KEY)