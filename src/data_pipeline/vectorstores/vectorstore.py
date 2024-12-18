import os
from dotenv import load_dotenv
from logger import logging

# Load environment variables
load_dotenv()


class VectorStoreProvider:
    def __init__(self, provider: str, collection_name: str, persist_directory: str = None, **kwargs):
        """
        Initialize the vector store provider.

        :param provider: Name of the vector store provider (e.g., "chroma", "pinecone").
        :param collection_name: Name of the vector store collection.
        :param persist_directory: Directory where the vector store data is persisted (only for Chroma).
        :param kwargs: Additional parameters required by specific providers.
        """
        self.provider = provider
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.kwargs = kwargs
        self.vectorstore = None  # Lazy initialization
        logging.info(f"VectorStoreProvider initialized for provider: {provider}")

    def _initialize_vectorstore(self, embedding_function):
        """
        Lazily initialize the vector store for the specified provider.
        """
        if self.provider == "chroma":
            self.vectorstore = self._initialize_chroma(embedding_function)
        elif self.provider == "pinecone":
            self.vectorstore = self._initialize_pinecone(embedding_function)
        else:
            raise ValueError(f"Unsupported vector store provider: {self.provider}")

    def _initialize_chroma(self, embedding_function):
        """
        Initialize the Chroma vector store with lazy imports.
        """
        logging.info("Initializing Chroma vector store...")
        try:
            from langchain_chroma import Chroma
        except ImportError:
            raise ImportError("Chroma is not installed. Install it with `pip install langchain[chromadb]`.")

        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=embedding_function,
        )

    def _initialize_pinecone(self, embedding_function):
        """
        Initialize the Pinecone vector store with lazy imports.
        """
        logging.info("Initializing Pinecone vector store...")
        try:
            import pinecone
            from langchain.vectorstores import Pinecone
        except ImportError:
            raise ImportError("Pinecone is not installed. Install it with `pip install pinecone-client`.")

        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        if not api_key or not environment:
            raise ValueError("Pinecone API key or environment not set in environment variables.")

        pinecone.init(api_key=api_key, environment=environment)
        index = pinecone.Index(self.collection_name)

        return Pinecone(index=index, embedding_function=embedding_function, text_key="text")

    def create_vector_store(self, documents, embedding_function):
        """
        Create and persist a vector store for the specified provider.

        :param documents: List of document objects.
        :param embedding_function: Embedding function to generate vector representations.
        """
        logging.info(f"Creating vector store with provider: {self.provider}")
        if self.provider == "chroma":
            try:
                from langchain_chroma import Chroma
            except ImportError:
                raise ImportError("Chroma is not installed. Install it with `pip install langchain[chromadb]`.")

            self.vectorstore = Chroma.from_documents(
                collection_name=self.collection_name,
                documents=documents,
                embedding=embedding_function,
                persist_directory=self.persist_directory,
            )
            logging.info(f"Chroma vector store persisted to '{self.persist_directory}'.")
        elif self.provider == "pinecone":
            try:
                import pinecone
                from langchain.vectorstores import Pinecone
            except ImportError:
                raise ImportError("Pinecone is not installed. Install it with `pip install pinecone-client`.")

            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
            index = pinecone.Index(self.collection_name)

            Pinecone.from_documents(
                documents=documents,
                embedding=embedding_function,
                index=index,
                text_key="text",
            )
            logging.info(f"Pinecone vector store created for collection: {self.collection_name}")
        else:
            raise ValueError(f"Vector store creation not supported for provider: {self.provider}")

    def get_retriever(self, embedding_function, k: int = 5):
        """
        Get a retriever from the vector store.

        :param embedding_function: Embedding function used for querying.
        :param k: Number of top results to retrieve.
        :return: Retriever instance.
        """
        if self.vectorstore is None:
            logging.info("Lazy initialization of vector store...")
            self._initialize_vectorstore(embedding_function)

        logging.info(f"Creating retriever for provider: {self.provider} with top {k} results.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    


if __name__ == "__main__":
    
    from src.data_pipeline.data_ingestion.document_loader import DocumentLoader
    from src.data_pipeline.data_processing.text_splitter import TextSplitter
    from src.data_pipeline.data_embeddings.embedding import EmbeddingProvider

    folder_path = r"C:\Users\rajes\PycharmProjects\chatraj\Data"

    # Load documents
    loader = DocumentLoader(folder_path)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents from the folder.")

    # Process documents with the specified chunking strategy
    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"Split the documents into {len(splits)} chunks.")

    # Initialize embedding provider
    embedding_provider = EmbeddingProvider(provider="openai")
    embeddings = embedding_provider.embed_documents([split.page_content for split in splits])

    # Use Chroma Vector Store
    try:
        collection_name = "my_collection"
        persist_directory = "./chroma_db"
        chroma_store = VectorStoreProvider(
            provider="chroma",
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        chroma_store.create_vector_store(splits, embedding_function=embedding_provider.embedder)
        print("Chroma vector store created successfully.")

        # Retrieve data from Chroma
        top_k = 2
        chroma_retriever = chroma_store.get_retriever(embedding_function=embedding_provider.embedder, k=top_k)
        query = "What is the use of AI-driven chatbot?"
        chroma_results = chroma_retriever.invoke(query)
        print("Chroma Results:")
        for i, result in enumerate(chroma_results):
            print(f"{i + 1}: {result.page_content}")
    except Exception as e:
        print(f"Error with Chroma vector store: {e}")

    # Use Pinecone Vector Store
    try:
        pinecone_store = VectorStoreProvider(
            provider="pinecone",
            collection_name=collection_name,
        )
        pinecone_store.create_vector_store(splits, embedding_function=embedding_provider.embedder)
        print("Pinecone vector store created successfully.")

        # Retrieve data from Pinecone
        pinecone_retriever = pinecone_store.get_retriever(embedding_function=embedding_provider.embedder, k=top_k)
        pinecone_results = pinecone_retriever.invoke(query)
        print("Pinecone Results:")
        for i, result in enumerate(pinecone_results):
            print(f"{i + 1}: {result.page_content}")
    except Exception as e:
        print(f"Error with Pinecone vector store: {e}")

