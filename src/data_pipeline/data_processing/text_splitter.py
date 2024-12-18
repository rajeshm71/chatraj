from logger import logging

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data_pipeline.data_ingestion.document_loader import DocumentLoader

class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the processor with a text splitter.
        :param chunk_size: Maximum size of each text chunk.
        :param chunk_overlap: Overlap size between chunks.
        """

        logging.info(f"Initializing DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks using the text splitter.
        """
        logging.info("Splitting documents into smaller chunks.")
        splits = self.text_splitter.split_documents(documents)
        logging.info(f"Documents split into {len(splits)} chunks.")
        return splits
    

# Main Execution
if __name__ == "__main__":
    folder_path = r"C:\Users\rajes\PycharmProjects\chatraj\Data"

    # Load documents
    loader = DocumentLoader(folder_path)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents from the folder.")

    # Process documents with the specified chunking strategy
    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"Split the documents into {len(splits)} chunks.")