import os
from dotenv import load_dotenv
from logger import logging

# Load environment variables
load_dotenv()

class EmbeddingProvider:
    def __init__(self, provider: str, **kwargs):
        """
        Initialize the embedding provider.

        :param provider: Name of the embedding provider (e.g., "openai", "huggingface").
        :param kwargs: Additional arguments required by the specific provider.
        """
        self.provider = provider
        self.embedder = None  # To be lazily initialized
        self.kwargs = kwargs  # Store additional arguments
        logging.info(f"EmbeddingProvider initialized for provider: {provider}")

    def _initialize_embedder(self):
        """
        Lazily initialize the embedding technique based on the provider.
        """
        if self.provider == "openai":
            self.embedder = self._load_openai_embeddings()
        elif self.provider == "huggingface":
            self.embedder = self._load_huggingface_embeddings(self.kwargs.get("model_name"))
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _load_openai_embeddings(self):
        """
        Load OpenAI embeddings.
        """
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        logging.info("Loading OpenAI embeddings...")
        return OpenAIEmbeddings(api_key=api_key)

    def _load_huggingface_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load Hugging Face embeddings.
        """
        if not model_name:
            raise ValueError("Model name for Hugging Face embeddings is required.")
        logging.info(f"Loading Hugging Face embeddings with model: {model_name}")
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return {"tokenizer": tokenizer, "model": model}
        except ImportError:
            raise ImportError(
                "Hugging Face transformers library is not installed. "
                "Install it with `pip install transformers`."
            )
        except Exception as e:
            raise RuntimeError(f"Error loading Hugging Face model: {e}")

    def embed_documents(self, documents):
        """
        Embed documents using the selected embedding technique.

        :param documents: List of documents to embed.
        :return: List of embeddings.
        """
        if self.embedder is None:
            logging.info("Lazy initialization of the embedder.")
            self._initialize_embedder()

        logging.info(f"Embedding {len(documents)} documents using {self.provider}...")

        if self.provider == "openai":
            # OpenAI embeddings
            embeddings = self.embedder.embed_documents(documents)
        elif self.provider == "huggingface":
            # Hugging Face embeddings
            tokenizer = self.embedder["tokenizer"]
            model = self.embedder["model"]
            embeddings = []
            for doc in documents:
                inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                # Use CLS token's embedding as representation
                embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy())
        else:
            raise ValueError(f"Embedding not supported for provider: {self.provider}")

        logging.info(f"Successfully created embeddings for {len(embeddings)} documents.")
        return embeddings


if __name__ == "__main__":
    from src.data_pipeline.data_ingestion.document_loader import DocumentLoader
    from src.data_pipeline.data_processing.text_splitter import TextSplitter

    folder_path = r"C:\Users\rajes\PycharmProjects\chatraj\Data"

    # Load documents
    loader = DocumentLoader(folder_path)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents from the folder.")

    # Process documents with the specified chunking strategy
    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"Split the documents into {len(splits)} chunks.")

    # Use OpenAI embeddings
    try:
        embedding_provider = EmbeddingProvider(provider="openai")
        openai_embeddings = embedding_provider.embed_documents([split.page_content for split in splits])
        print(f"Generated {len(openai_embeddings)} embeddings using OpenAI.")
    except Exception as e:
        print(f"Error with OpenAI embeddings: {e}")

    # Use Hugging Face embeddings
    try:
        embedding_provider = EmbeddingProvider(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")
        hf_embeddings = embedding_provider.embed_documents([split.page_content for split in splits])
        print(f"Generated {len(hf_embeddings)} embeddings using Hugging Face.")
    except Exception as e:
        print(f"Error with Hugging Face embeddings: {e}")
