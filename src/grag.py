from typing import Any

from src.query_engine import QueryEngine

class GraphRAG(QueryEngine):

    def __init__(self, **kwargs : Any):
        """
        Initializes the GraphRAG system with components for document processing, knowledge graph construction,
        querying, and visualization.

        Attributes:
        - llm: An instance of a large language model (LLM) for generating responses.
        - embedding_model: An instance of an embedding model for document embeddings.
        - document_processor: An instance of the DocumentProcessor class for processing documents.
        - knowledge_graph: An instance of the KnowledgeGraph class for building and managing the knowledge graph.
        - query_engine: An instance of the QueryEngine class for handling queries (initialized as None).
        - visualizer: An instance of the Visualizer class for visualizing the knowledge graph traversal.
        """
        super().__init__(**kwargs)

        # TODO: Initialize the neojs graph and save it in the knowledge graph