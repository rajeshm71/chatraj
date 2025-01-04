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
        self.prompt = self.prompt_manager.get_context_based_question_answering_prompt(context="{context}", question="{question}")

        logging.info("RAG chain components initialized.")

    @staticmethod
    def docs_to_str(documents):
        """
        Convert a list of documents to a single string for context.

        :param documents: List of retrieved documents.
        :return: A single string combining all document contents.
        """
        return " ".join([doc.page_content for doc in documents])

    def run(self, question: str):
        """
        Run the RAG chain to generate a response for a given question.

        :param question: User query or question.
        :return: Generated response from the RAG chain.
        """
        if not all([self.retriever, self.prompt, self.llm]):
            raise ValueError("RAG chain components are not initialized. Call `initialize_chain()` first.")

        # Define the chain
        rag_chain = (
            {"context": self.retriever | self.docs_to_str, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Run the chain and return the response
        logging.info(f"Running RAG chain for question: {question}")
        response = rag_chain.invoke(question)
        return response
    
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
    response = rag_chain.run(question)
    print("Response:", response)

