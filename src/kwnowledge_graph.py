import logging
import spacy
from typing import List, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from nltk.stem import WordNetLemmatizer
from spacy.cli import download
from src.documents_processor import DocumentProcessor, compute_similarity_matrix

# Configura il logger per il modulo
logger = logging.getLogger(__name__)

# Definisce la classe Concepts per rappresentare una lista di concetti
class Concepts(BaseModel):
    """
    Pydantic model to represent a list of concepts.

    Attributes:
        concepts_list (List[str]): List of concepts.
    """
    concepts_list: List[str] = Field(description="List of concepts")

# Definisce la classe KnowledgeGraph per costruire e gestire un grafo della conoscenza
class KnowledgeGraph(DocumentProcessor):
    """
    Class to build and manage a knowledge graph based on documents.

    Attributes:
        lemmatizer (WordNetLemmatizer): Lemmatizer for processing concepts.
        concept_cache (dict): Cache to store extracted concepts.
        nlp (spacy.Language): spaCy NLP model.
        edges_threshold (float): Threshold for adding edges based on similarity.
        spacy_models (list[str]): List of spaCy models to load.
    """
    _graph = None
    lemmatizer = None
    concept_cache = None
    nlp = None
    edges_threshold: float = 0.8
    spacy_models: list[str] = ["en_core_web_sm"]

    model: str = "gpt-4o-mini"
    max_token: int = 4000
    temperature: float = 0
    llm: Any = None
    max_context_length: int = 4000

    def __init__(self, **kwargs):
        """
        Initializes an instance of KnowledgeGraph.

        Args:
            **kwargs: Optional arguments to configure the instance.
        """
        super().__init__(**kwargs)
        self._graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = kwargs.get("concept_cache", {})
        self._load_spacy_model()
        self.edges_threshold = kwargs.get("edges_threshold", 0.8)

        self.temperature = kwargs.get("temperature", 0)
        self.model = kwargs.get("model", "gpt-4o-mini")
        self.max_token = kwargs.get("max_token", 4000)
        self.max_context_length = kwargs.get("max_context_length", 4000)

        self.llm = ChatOpenAI(temperature=0,
                              model=self.model,
                              max_tokens=self.max_token)

    def _create_embeddings(self, splits):
        """
        Creates embeddings for document fragments using the embedding model.

        Args:
            splits (list): List of document fragments.

        Returns:
            numpy.ndarray: Array of embeddings for the document fragments.
        """
        texts = [split.page_content for split in splits]
        return self.embedchain.embed_documents(texts)

    def _build_graph(self, splits):
        """
        Builds the knowledge graph by adding nodes, creating embeddings, extracting concepts, and adding edges.

        Args:
            splits (list): List of document fragments.

        Returns:
            None
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits)
        self._extract_concepts(splits, self.llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        Adds nodes to the graph from document fragments.

        Args:
            splits (list): List of document fragments.

        Returns:
            None
        """
        for i, split in enumerate(splits):
            self._graph.add_node(i, content=split.page_content)

    def _load_spacy_model(self):
        """
        Loads the spaCy NLP model, downloading it if necessary.

        Returns:
            None
        """
        try:
            for model in self.spacy_models:
                if not spacy.util.is_package(model):
                    print(f"Downloading spaCy model: {model}...")
                    download(model)
        except OSError:
            logger.error("Error downloading spaCy model. Please check your internet connection.")
            raise OSError

    def _extract_concepts_and_entities(self, content, llm):
        """
        Extracts concepts and named entities from the content using spaCy and a language model.

        Args:
            content (str): Content from which to extract concepts and entities.
            llm: Instance of a large language model.

        Returns:
            list: List of extracted concepts and entities.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Estrae entità nominate usando spaCy
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # Estrae concetti generali usando il modello di linguaggio
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        # Combina entità nominate e concetti generali
        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        Extracts concepts for all document fragments using multi-threading.

        Args:
            splits (list): List of document fragments.
            llm: Instance of a large language model.

        Returns:
            None
        """
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total=len(splits),
                               desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self._graph.nodes[node]['concepts'] = concepts

    def _add_edges(self, embeddings):
        """
        Adds edges to the graph based on the similarity of embeddings and shared concepts.

        Args:
            embeddings (numpy.ndarray): Array of embeddings for the document fragments.

        Returns:
            None
        """
        similarity_matrix = compute_similarity_matrix(embeddings)
        num_nodes = len(self._graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self._graph.nodes[node1]['concepts']) & set(
                        self._graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self._graph.add_edge(node1, node2, weight=edge_weight,
                                         similarity=similarity_score,
                                         shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        Calculates the weight of an edge based on the similarity score and shared concepts.

        Args:
            node1 (int): First node.
            node2 (int): Second node.
            similarity_score (float): Similarity score between the nodes.
            shared_concepts (set): Set of shared concepts between the nodes.
            alpha (float, optional): Weight of the similarity score. Default is 0.7.
            beta (float, optional): Weight of the shared concepts. Default is 0.3.

        Returns:
            float: Calculated weight of the edge.
        """
        max_possible_shared = min(len(self._graph.nodes[node1]['concepts']), len(self._graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        Lemmatizes a given concept.

        Args:
            concept (str): Concept to be lemmatized.

        Returns:
            str: Lemmatized concept.
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])