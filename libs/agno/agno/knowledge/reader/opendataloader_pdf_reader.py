import asyncio
import shutil
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union
from uuid import uuid4

from agno.knowledge.chunking.document import DocumentChunking
from agno.knowledge.chunking.strategy import ChunkingStrategy, ChunkingStrategyType
from agno.knowledge.document.base import Document
from agno.knowledge.reader.base import Reader
from agno.knowledge.types import ContentType
from agno.utils.log import log_debug, log_error

try:
    import opendataloader_pdf
except ImportError:
    raise ImportError(
        "The `opendataloader-pdf` package is not installed. Please install it via `pip install opendataloader-pdf`."
    )

VALID_OUTPUT_FORMATS = ("markdown", "text", "html", "json")


class OpenDataLoaderPDFReader(Reader):
    """Reader for PDF files using the opendataloader-pdf parser.

    opendataloader-pdf extracts layout-aware content (reading order, tables, headings)
    from PDFs and exports it as markdown, text, html or json. It runs a Java process
    per document, so a Java runtime (11 or newer) must be available on the PATH.
    """

    def __init__(
        self,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        output_format: str = "markdown",
        password: Optional[str] = None,
        pages: Optional[str] = None,
        timeout: Optional[float] = None,
        convert_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the OpenDataLoaderPDFReader.

        Args:
            chunking_strategy: Strategy to use for chunking the documents.
            output_format: Output format of the parser. One of "markdown" (default), "text", "html" or "json".
            password: Default password for encrypted PDFs. Overridden by the ``password`` argument of ``read``.
            pages: Pages to extract, for example "1,3,5-7". Defaults to all pages.
            timeout: Wall-clock limit in seconds for each conversion. Defaults to no limit.
            convert_options: Extra options passed to ``opendataloader_pdf.convert_bytes``,
                for example ``{"table_method": "cluster", "include_header_footer": True}``.
            **kwargs: Additional arguments passed to the Reader class.
        """
        if chunking_strategy is None:
            chunk_size = kwargs.get("chunk_size", 5000)
            chunking_strategy = DocumentChunking(chunk_size=chunk_size)
        super().__init__(chunking_strategy=chunking_strategy, **kwargs)

        output_format = output_format.lower()
        if output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(f"Invalid output format: '{output_format}'. Valid options: {list(VALID_OUTPUT_FORMATS)}")

        self.output_format = output_format
        self.password = password
        self.pages = pages
        self.timeout = timeout
        self.convert_options: Dict[str, Any] = dict(convert_options or {})

    @classmethod
    def get_supported_chunking_strategies(cls) -> List[ChunkingStrategyType]:
        """Get the list of supported chunking strategies for OpenDataLoader PDF readers."""
        return [
            ChunkingStrategyType.DOCUMENT_CHUNKER,
            ChunkingStrategyType.CODE_CHUNKER,
            ChunkingStrategyType.FIXED_SIZE_CHUNKER,
            ChunkingStrategyType.AGENTIC_CHUNKER,
            ChunkingStrategyType.SEMANTIC_CHUNKER,
            ChunkingStrategyType.RECURSIVE_CHUNKER,
        ]

    @classmethod
    def get_supported_content_types(cls) -> List[ContentType]:
        return [ContentType.PDF]

    @staticmethod
    def _get_doc_name(pdf: Union[str, Path, IO[Any]], name: Optional[str] = None) -> str:
        if name:
            return name
        if isinstance(pdf, (str, Path)):
            return Path(pdf).stem.replace(" ", "_")
        return getattr(pdf, "name", "pdf_file").split(".")[0].replace(" ", "_")

    @staticmethod
    def _load_bytes(pdf: Union[str, Path, IO[Any]]) -> bytes:
        if isinstance(pdf, (str, Path)):
            path = Path(pdf)
            if not path.exists():
                raise FileNotFoundError(f"Could not find file: {path}")
            return path.read_bytes()
        data = pdf.read()
        if isinstance(data, str):
            raise ValueError("Unsupported file object: expected a binary stream.")
        return data

    def _convert(self, data: bytes, password: Optional[str]) -> str:
        if shutil.which("java") is None:
            raise RuntimeError(
                "opendataloader-pdf needs a Java runtime (11 or newer) on the PATH, but `java` was not found."
            )

        options: Dict[str, Any] = {"quiet": True, **self.convert_options}
        if password or self.password:
            options["password"] = password or self.password
        if self.pages is not None:
            options["pages"] = self.pages
        if self.timeout is not None:
            options["timeout"] = self.timeout

        return opendataloader_pdf.convert_bytes(data, format=self.output_format, **options)

    def _build_documents(self, content: str, doc_name: str) -> List[Document]:
        if not content or not content.strip():
            log_error(f"No content extracted from PDF: {doc_name}")
            return []

        document = Document(name=doc_name, id=str(uuid4()), content=content)
        if self.chunk:
            return self.chunk_document(document)
        return [document]

    def read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        """Read a PDF with opendataloader-pdf and return its documents."""
        if pdf is None:
            log_error("No pdf provided")
            return []

        doc_name = self._get_doc_name(pdf, name)
        log_debug(f"Reading: {doc_name}")

        try:
            data = self._load_bytes(pdf)
            content = self._convert(data, password)
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            log_error(f"Error reading PDF with opendataloader-pdf: {doc_name}: {str(e)}")
            return []

        return self._build_documents(content, doc_name)

    async def async_read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
    ) -> List[Document]:
        """Asynchronously read a PDF with opendataloader-pdf and return its documents."""
        return await asyncio.to_thread(self.read, pdf, name, password)
