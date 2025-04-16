import os
import uuid
import nltk
from typing import Any, List
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from langchain_core.documents import Document
from langchain.vectorstores import FAISS

class VectorDB:
    """
    A class to manage interactions with a Qdrant vector database.

    Attributes:
        url (str): The URL of the Qdrant server.
        port (int): The port of the Qdrant server.
        collection_name (str): The name of the collection in the Qdrant database.
        client (QdrantClient): The Qdrant client instance.
        embedchain (OpenAIEmbeddings): The embedding model used for vectorization.
        dimension (int): The dimensionality of the vectors.
        distance (Distance): The distance metric used for vector similarity.
        vector (QdrantVectorStore): The vector store instance.
    """

    url: str
    port: int
    collection_name: str
    client: QdrantClient = None
    embedchain: OpenAIEmbeddings
    dimension: int = 1536
    distance: Distance = Distance.COSINE
    vector: QdrantVectorStore = None
    _vs: Any = None
    type : str = None
    nltk_identifier: list[str] = ["punkt", "stopwords", "wordnet"]

    def __init__(self, **kwargs):
        """
        Initializes the VectorDB instance.

        Args:
            **kwargs: Optional keyword arguments to override default attributes.
                - url (str): The Qdrant server URL.
                - port (int): The Qdrant server port.
                - embedchain (OpenAIEmbeddings): The embedding model.
                - collection_name (str): The collection name.
                - dimension (int): The vector dimensionality.
                - distance (Distance): The distance metric.
        """
        self.url = kwargs.get("url", os.getenv('QDRANT_URL'))
        self.port = kwargs.get("port", os.getenv('QDRANT_PORT', 6333))
        self.embedchain = kwargs.get("embedchain", OpenAIEmbeddings(model="text-embedding-3-small"))
        self.collection_name = kwargs.get("collection_name", f"graph_rag_{os.getenv("APP_ENV", "development")}")
        self.dimension = kwargs.get("dimension", 1536)
        self.distance = kwargs.get("distance", Distance.COSINE)
        self.type = kwargs.get("type", "vector")
        if self.type == "vector":
            self.client = self._connection()
            self._vector = self._vector_store(**kwargs)
        self._init_nltk()

    def _init_nltk(self):
        """
        Downloads the required NLTK resources specified in the nltk_identifier attribute.

        Raises:
            LookupError: If an NLTK resource cannot be downloaded.
        """
        for identifier in self.nltk_identifier:
            try:
                nltk.download(identifier)
            except LookupError:
                nltk.download(identifier)

    def _connection(self):
        """
        Establishes a connection to the Qdrant server.

        Returns:
            QdrantClient: The Qdrant client instance.

        Raises:
            Exception: If the QDRANT_URL or QDRANT_PORT environment variables are not set.
        """
        if self.url is None or self.url == "":
            raise Exception("The QDRANT_URL environment variable must be provided")

        if self.port is None or self.port == "":
            raise Exception("The QDRAND_PORT environment variable must be provided")

        return QdrantClient(url=self.url, port=self.port)

    def _vector_store(self, **kwargs: Any) -> QdrantVectorStore:
        """
        Initializes or retrieves a Qdrant vector store.

        Args:
            **kwargs: Optional keyword arguments to override default attributes.
                - collection (str): The collection name.
                - dimension (int): The vector dimensionality.
                - distance (Distance): The distance metric.
                - embedchain (OpenAIEmbeddings): The embedding model.

        Returns:
            QdrantVectorStore: The initialized vector store.
        """
        collection = kwargs.get("collection", self.collection_name).lower()
        dimension = kwargs.get("dimension", self.dimension)
        distance = kwargs.get("distance", self.distance)
        embedchain = kwargs.get("embedchain", self.embedchain)

        collections_list = self.client.get_collections()
        existing_collections = [col.name for col in collections_list.collections]

        if collection not in existing_collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=distance)
            )

        return QdrantVectorStore.from_existing_collection(
            embedding=embedchain,
            collection_name=collection,
            url=self.url
        )

    def add(self, documents: List[Document], **kwargs: Any):
        """
        Adds documents to the vector store.

        Args:
            documents (List[Document]): A list of documents to add.
            **kwargs: Optional keyword arguments to override default attributes.
                - vs (QdrantVectorStore): The vector store instance.
        """
        if self.type == "vector":
            self._vs = kwargs.get("vs", self.vector)
            new_documents = []
            for doc in documents:
                d = Document(
                    page_content=doc.page_content,
                    id=str(uuid.uuid4())
                )
                new_documents.append(d)
            self._vs.add_documents(new_documents)
        else:
            self._vs = FAISS.from_documents(documents, self.embedchain)

