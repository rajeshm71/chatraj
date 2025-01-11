from src.data_pipeline.data_ingestion.document_loader import DocumentLoader
from src.data_pipeline.data_processing.text_splitter import TextSplitter
from src.data_pipeline.data_embeddings.embedding import EmbeddingProvider
from src.data_pipeline.vectorstores.vectorstore import VectorStoreProvider
from src.generation_pipeline.models.model import ModelProvider
from src.generation_pipeline.prompts.prompt_manager import PromptManager
from utils import docs2str
from logger import logging

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage


class RAGChain:
    def __init__(self, config_path: str = None):
        """
        Initialize the RAGChain with a configuration dictionary.

        :param config: Configuration dictionary for RAGChain.
        """
        self.config_path = config_path
        self.document_loader = None
        self.text_splitter = None
        self.embedding_provider = None
        self.vectorstore_provider = None
        self.model_provider = None
        self.prompt_manager = None
        self.chat_history = []

        # Lazy-initialized components
        self.retriever = None
        self.llm = None
        self.prompt = None

    def initialize_chain(self, data_path: str = None):
        """
        Initialize the RAG chain components.

        :param uploaded_data_path: Path to the uploaded data for processing.
        """
        # Initialize components
        self.document_loader = DocumentLoader(data_path)
        self.text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_provider = EmbeddingProvider(provider='openai')
        self.vectorstore_provider = VectorStoreProvider(provider='chroma', collection_name="my_collection", persist_directory="./chroma_db")
        self.model_provider = ModelProvider(config_path=self.config_path)
        self.prompt_manager = PromptManager()

        # Process documents
        documents = self.document_loader.load_documents()
        splits = self.text_splitter.split_documents(documents)

        # Embedding and vector store
        embeddings = self.embedding_provider.embed_documents([split.page_content for split in splits])
        self.vectorstore_provider.create_vector_store(splits, self.embedding_provider.embedder)
        #retriever_k = self.config["vectorstore"].get("retriever_k", 3)
        self.retriever = self.vectorstore_provider.get_retriever(self.embedding_provider.embedder, k=5)

        # Load model and prompt
        #model_name = self.config["model"]["default_provider"]
        self.llm = self.model_provider.switch_provider('openai_gpt4')
        self.context_based_qa_prompt = self.prompt_manager.get_context_based_question_answering_prompt(context="{context}", question="{question}")
        self.contextualize_question_prompt = self.prompt_manager.get_contextualize_question_prompt(chat_history_placeholder="chat_history", user_input="{input}")
        self.qa_prompt  = self.prompt_manager.get_qa_prompt(context="{context}", chat_history_placeholder="chat_history, {input}")

        logging.info("RAG chain components initialized.")

    @staticmethod
    def docs_to_str(documents):
        """
        Convert a list of documents to a single string for context.

        :param documents: List of retrieved documents.
        :return: A single string combining all document contents.
        """
        return " ".join([doc.page_content for doc in documents])
    

    def update_chat_history(self, user_input: str, model_response: str):
        """
        Update the chat history with the latest user input and model response.

        :param user_input: User's input or question.
        :param model_response: Model's response to the user's input.
        """
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=model_response)
        ])
    
    def contextualize_question(self, user_input: str):
        """
        Contextualize a user input question based on chat history.

        :param user_input: User's input or follow-up question.
        :return: Reformulated standalone question.
        """
        if not self.contextualize_question_prompt or not self.llm:
            raise ValueError("Contextualization components are not initialized. Call `initialize_chain()` first.")

        contextualize_chain = self.contextualize_question_prompt | self.llm | StrOutputParser()
        standalone_question = contextualize_chain.invoke({"input": user_input, "chat_history": self.chat_history})
        logging.info(f"Contextualized question: {standalone_question}")
        return standalone_question
    

    def run_with_chat_history(self):
        """
        Run the RAG chain with chat history for a given question.
        :return: Generated response from the RAG chain.
        """
        if not self.contextualize_question_prompt or not self.llm or not self.retriever:
            raise ValueError("RAG chain components are not initialized. Call `initialize_chain()` first.")

        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_question_prompt)
        

        # Create question-answer chain
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        # Create full RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Run the chain with chat history
        # model_response = ""
        # for chunk in rag_chain.stream({"input": user_input, "chat_history": self.chat_history}):
        #     print(chunk.get('answer', ''), end=" ", flush=True)
        #     model_response += chunk.get('answer', '')

        # Update chat history
        #self.update_chat_history(user_input=user_input, model_response=model_response)

        return rag_chain



    def run(self, question: str):
        """
        Run the RAG chain to generate a response for a given question.

        :param question: User query or question.
        :return: Generated response from the RAG chain.
        """
        if not all([self.retriever, self.context_based_qa_prompt, self.llm]):
            raise ValueError("RAG chain components are not initialized. Call `initialize_chain()` first.")

        # Define the chain
        rag_chain = (
            {"context": self.retriever | self.docs_to_str, "question": RunnablePassthrough()}
            | self.context_based_qa_prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain and return the response
        logging.info(f"Running RAG chain for question: {question}")
        response = rag_chain.invoke(question)
        return response, rag_chain
    
if __name__ == "__main__":
    # Path to the unified config.yaml
    config_path = "config/config.yaml"
    data_path = "Data"
    # Initialize the RAGChain
    rag_chain = RAGChain(config_path)
    rag_chain.initialize_chain(data_path=data_path)

    # Input question
    question = "What is the use of AI-driven chatbot?"

    # Run the chain and get the response
    response, retrieval_chain = rag_chain.run(question)
    print("Response:", response)
    for chunk in retrieval_chain.stream(question):
        print(chunk, end=" ", flush=True)


    question_1 = "What is the use of AI-driven chatbot?"
    response_1, retrieval_chain = rag_chain.run_with_chat_history(question_1)
    print(f"Q: {question_1}")
    print(f"A: {response_1}")

    # Ask a follow-up question
    question_2 = "Where are its benefits?"
    response_2, retrieval_chain = rag_chain.run_with_chat_history(question_2)
    print(f"Q: {question_2}")
    print(f"A: {response_2}")

