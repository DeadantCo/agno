import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from agno.knowledge.reader.opendataloader_pdf_reader import OpenDataLoaderPDFReader
from agno.knowledge.reader.reader_factory import ReaderFactory
from agno.knowledge.types import ContentType

MODULE = "agno.knowledge.reader.opendataloader_pdf_reader"
PDF_BYTES = b"%PDF-1.4 fake"


@pytest.fixture
def mock_convert():
    with (
        patch(f"{MODULE}.shutil.which", return_value="/usr/bin/java"),
        patch(f"{MODULE}.opendataloader_pdf.convert_bytes", return_value="# Title\n\nSome body text.") as mocked,
    ):
        yield mocked


def test_read_path(tmp_path: Path, mock_convert):
    pdf = tmp_path / "my report.pdf"
    pdf.write_bytes(PDF_BYTES)

    documents = OpenDataLoaderPDFReader().read(pdf)

    assert len(documents) == 1
    assert documents[0].name == "my_report"
    assert "Some body text." in documents[0].content
    args, kwargs = mock_convert.call_args
    assert args[0] == PDF_BYTES
    assert kwargs["format"] == "markdown"
    assert kwargs["quiet"] is True
    assert "password" not in kwargs


def test_read_str_path_and_name_override(tmp_path: Path, mock_convert):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_BYTES)

    documents = OpenDataLoaderPDFReader().read(str(pdf), name="custom")

    assert documents[0].name == "custom"


def test_read_bytesio(mock_convert):
    stream = BytesIO(PDF_BYTES)
    stream.name = "upload.pdf"  # type: ignore[attr-defined]

    documents = OpenDataLoaderPDFReader().read(stream)

    assert documents[0].name == "upload"
    assert mock_convert.call_args.args[0] == PDF_BYTES


def test_read_missing_file_raises(mock_convert):
    with pytest.raises(FileNotFoundError):
        OpenDataLoaderPDFReader().read(Path("does_not_exist.pdf"))


def test_read_none_returns_empty(mock_convert):
    assert OpenDataLoaderPDFReader().read(None) == []


def test_options_are_forwarded(mock_convert):
    reader = OpenDataLoaderPDFReader(
        output_format="TEXT",
        password="default",
        pages="1-2",
        timeout=30,
        convert_options={"table_method": "cluster"},
    )
    reader.read(BytesIO(PDF_BYTES), name="doc", password="override")

    kwargs = mock_convert.call_args.kwargs
    assert kwargs["format"] == "text"
    assert kwargs["password"] == "override"
    assert kwargs["pages"] == "1-2"
    assert kwargs["timeout"] == 30
    assert kwargs["table_method"] == "cluster"


def test_constructor_password_is_default(mock_convert):
    OpenDataLoaderPDFReader(password="secret").read(BytesIO(PDF_BYTES), name="doc")

    assert mock_convert.call_args.kwargs["password"] == "secret"


def test_invalid_output_format():
    with pytest.raises(ValueError, match="Invalid output format"):
        OpenDataLoaderPDFReader(output_format="yaml")


def test_chunking_can_be_disabled(mock_convert):
    mock_convert.return_value = "para one\n\n" + "x" * 200
    documents = OpenDataLoaderPDFReader(chunk=False).read(BytesIO(PDF_BYTES), name="doc")

    assert len(documents) == 1
    assert documents[0].content == mock_convert.return_value


def test_chunking_splits_long_content(mock_convert):
    mock_convert.return_value = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    documents = OpenDataLoaderPDFReader(chunk_size=500).read(BytesIO(PDF_BYTES), name="doc")

    assert len(documents) > 1


def test_empty_output_returns_empty(mock_convert):
    mock_convert.return_value = "  \n"

    assert OpenDataLoaderPDFReader().read(BytesIO(PDF_BYTES), name="doc") == []


def test_parser_failure_returns_empty(mock_convert):
    mock_convert.side_effect = RuntimeError("boom")

    assert OpenDataLoaderPDFReader().read(BytesIO(PDF_BYTES), name="doc") == []


def test_missing_java_returns_empty_without_calling_parser():
    with (
        patch(f"{MODULE}.shutil.which", return_value=None),
        patch(f"{MODULE}.opendataloader_pdf.convert_bytes") as mocked,
    ):
        assert OpenDataLoaderPDFReader().read(BytesIO(PDF_BYTES), name="doc") == []
        mocked.assert_not_called()


def test_async_read(mock_convert):
    documents = asyncio.run(OpenDataLoaderPDFReader().async_read(BytesIO(PDF_BYTES), name="doc"))

    assert len(documents) == 1
    assert documents[0].name == "doc"


def test_supported_types_and_strategies():
    assert OpenDataLoaderPDFReader.get_supported_content_types() == [ContentType.PDF]
    assert OpenDataLoaderPDFReader.get_supported_chunking_strategies()


def test_factory_registration():
    assert ReaderFactory.get_reader_class("opendataloader_pdf") is OpenDataLoaderPDFReader
    assert "opendataloader_pdf" in ReaderFactory.get_all_reader_keys()
    assert "opendataloader_pdf" in ReaderFactory.READER_METADATA

    reader = ReaderFactory.create_reader("opendataloader_pdf", output_format="text")
    assert isinstance(reader, OpenDataLoaderPDFReader)
    assert reader.output_format == "text"
