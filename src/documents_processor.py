import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from src.vector_store import VectorDB


def compute_similarity_matrix(embeddings):
    """
    Computes a cosine similarity matrix for a given set of embeddings.

    Args:
    - embeddings (numpy.ndarray): An array of embeddings.

    Returns:
    - numpy.ndarray: A cosine similarity matrix for the input embeddings.
    """
    return cosine_similarity(embeddings)


class DocumentProcessor(VectorDB):

    chunk_size : int = 1000
    chunk_overlap : int = 200
    text_splitter : RecursiveCharacterTextSplitter

    def __init__(self, **kwargs):
        """
        Initializes the DocumentProcessor with a text splitter and OpenAI embeddings.

        Attributes:
        - text_splitter: An instance of RecursiveCharacterTextSplitter with specified chunk size and overlap.
        - embeddings: An instance of OpenAIEmbeddings used for embedding documents.
        """
        super().__init__(**kwargs)
        self.chunk_size = kwargs.get("chunk_size", 1000)
        self.chunk_overlap = kwargs.get("chunk_overlap", 200)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                  chunk_overlap=self.chunk_overlap)

    def _process(self, documents):
        """
        Processes a list of documents by splitting them into smaller chunks and creating a vector store.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - tuple: A tuple containing:
          - splits (list of str): The list of split document chunks.
          - vector_store (FAISS): A FAISS vector store created from the split document chunks and their embeddings.
        """
        splits = self.text_splitter.split_documents(documents)
        self.add(splits)
        return splits

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        Creates embeddings for a list of texts in batches.

        Args:
        - texts (list of str): A list of texts to be embedded.
        - batch_size (int, optional): The number of texts to process in each batch. Default is 32.

        Returns:
        - numpy.ndarray: An array of embeddings for the input texts.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedchain.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)