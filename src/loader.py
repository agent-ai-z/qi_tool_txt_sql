from abc import abstractmethod, ABC
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader
from src.grag import GraphRAG

class Loader:
    graph_rag = None
    file_path: str

    def __init__(self, file_path: str, **kwargs):
        """
        Initializes the Loader with a GraphRAG instance.

        Args:
        - kwargs: Optional keyword arguments to override default attributes.
        """
        self.file_path = file_path
        self.graph_rag = GraphRAG(**kwargs)

    @abstractmethod
    def load(self):
        pass

class LoaderPDF(Loader, ABC):

    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)

    def load(self):
        """
        Loads a PDF file and returns its content.

        Returns:
        - str: The content of the PDF file.
        """
        loader = PyPDFLoader(self.file_path)
        return loader.load()

class LoaderJSON(Loader, ABC):

    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)

    def load(self):
        """
        Loads a JSON file and returns its content.

        Returns:
        - dict: The content of the JSON file.
        """
        loader = JSONLoader(file_path=self.file_path, jq_schema=".",
                            text_content=False)  # Usa jq_schema per specificare come estrarre i dati
        return loader.load()

class LoaderMD(Loader, ABC):

    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)

    def load(self):
        """
        Loads a Markdown file and returns its content.

        Returns:
        - str: The content of the Markdown file.
        """
        loader = UnstructuredMarkdownLoader(self.file_path)
        return loader.load()