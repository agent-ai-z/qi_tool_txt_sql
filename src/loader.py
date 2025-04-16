from abc import abstractmethod, ABC
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader
from src.grag import GraphRAG

class Loader(GraphRAG, ABC):
    """
    Abstract base class for file loaders that extends the GraphRAG class.

    Attributes:
        file_path (str): Path to the file to be loaded.
        loader (Any): Instance of the loader used to process the file.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initializes the Loader with a GraphRAG instance.

        Args:
            file_path (str): Path to the file to be loaded.
            **kwargs: Optional keyword arguments to override default attributes.
        """
        super().__init__(**kwargs)
        self.file_path = file_path

    @abstractmethod
    def load(self):
        """
        Abstract method to load the file content.
        Must be implemented by subclasses.
        """
        pass

    def process(self):
        """
        Processes the loaded file content by splitting it into fragments
        and building a graph representation.

        Steps:
            1. Load the file content using the `load` method.
            2. Process the content into fragments using `_process`.
            3. Build a graph from the fragments using `_build_graph`.
        """
        l = self.load()
        splits = self._process(l)
        self._build_graph(splits)

class LoaderPDF(Loader, ABC):
    """
    Loader class for processing PDF files.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initializes the LoaderPDF with the file path and optional arguments.

        Args:
            file_path (str): Path to the PDF file to be loaded.
            **kwargs: Optional keyword arguments to override default attributes.
        """
        super().__init__(file_path, **kwargs)

    def load(self):
        """
        Loads a PDF file and returns its content.

        Returns:
            str: The content of the PDF file.
        """
        loader = PyPDFLoader(self.file_path)
        return loader.load()

class LoaderJSON(Loader, ABC):
    """
    Loader class for processing JSON files.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initializes the LoaderJSON with the file path and optional arguments.

        Args:
            file_path (str): Path to the JSON file to be loaded.
            **kwargs: Optional keyword arguments to override default attributes.
        """
        super().__init__(file_path, **kwargs)

    def load(self):
        """
        Loads a JSON file and returns its content.

        Returns:
            dict: The content of the JSON file.
        """
        loader = JSONLoader(file_path=self.file_path, jq_schema=".",
                            text_content=False)  # Uses jq_schema to specify how to extract data.
        return loader.load()

class LoaderMD(Loader, ABC):
    """
    Loader class for processing Markdown files.
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initializes the LoaderMD with the file path and optional arguments.

        Args:
            file_path (str): Path to the Markdown file to be loaded.
            **kwargs: Optional keyword arguments to override default attributes.
        """
        super().__init__(file_path, **kwargs)

    def load(self):
        """
        Loads a Markdown file and returns its content.

        Returns:
            str: The content of the Markdown file.
        """
        loader = UnstructuredMarkdownLoader(self.file_path)
        return loader.load()