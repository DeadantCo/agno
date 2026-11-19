"""
OpenDataLoader PDF Reader
=========================
Parse PDFs with opendataloader-pdf, a layout-aware parser that keeps reading order,
tables and headings. Useful when the default PDFReader flattens complex layouts.

Requirements:
- `uv pip install opendataloader-pdf`
- A Java runtime (11 or newer) available on the PATH

Output formats: "markdown" (default), "text", "html", "json".

See also: 01_documents.py for the default PDF reader.
"""

import asyncio

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.opendataloader_pdf_reader import OpenDataLoaderPDFReader
from agno.models.openai import OpenAIResponses
from agno.vectordb.lancedb import LanceDb, SearchType

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

knowledge = Knowledge(
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="opendataloader_pdf",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    async def main():
        await knowledge.ainsert(
            name="CV",
            path="cookbook/07_knowledge/testing_resources/cv_1.pdf",
            reader=OpenDataLoaderPDFReader(output_format="markdown"),
        )
        agent.print_response("What skills does Jordan Mitchell have?", stream=True)

    asyncio.run(main())
