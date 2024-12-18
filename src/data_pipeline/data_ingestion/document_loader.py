from logger import logging

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

class DocumentLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.documents = []

    def load_documents(self) -> List[Document]:
        """
        Load documents from the specified folder.
        """

        logging.info(f"Loading documents from folder: {self.folder_path}")
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                logging.warning(f"Unsupported file type: {filename}")
                continue
            loaded_docs = loader.load()
            logging.debug(f"Loaded {len(loaded_docs)} documents from {filename}")
            self.documents.extend(loaded_docs)
        logging.info(f"Total documents loaded: {len(self.documents)}")
        return self.documents
    
if __name__ == "__main__":
    folder_path = r"C:\Users\rajes\PycharmProjects\chatraj\Data"

    # Load documents
    loader = DocumentLoader(folder_path)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents from the folder.")