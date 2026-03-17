"""Document parsing utilities."""
from src.models.document import Document
from typing import List
import os
from hashlib import md5


def parse_markdown(content: str, source: str) -> Document:
    """Parse markdown content into a Document.

    Args:
        content: The markdown content as string
        source: Source identifier (file path, URL, etc)

    Returns:
        Document object
    """
    # Extract title from first heading or use filename
    title = extract_title_from_markdown(content, source)

    doc_id = md5(f"{source}:{title}".encode()).hexdigest()

    metadata = {
        "file_type": "markdown",
    }

    return Document(
        id=doc_id,
        content=content,
        title=title,
        source=source,
        source_type="file",
        metadata=metadata,
    )


def parse_text(content: str, source: str) -> Document:
    """Parse plain text content into a Document.

    Args:
        content: The text content as string
        source: Source identifier (file path, URL, etc)

    Returns:
        Document object
    """
    # Extract title from first line or use filename
    title = extract_title_from_text(content, source)

    doc_id = md5(f"{source}:{title}".encode()).hexdigest()

    metadata = {
        "file_type": "text",
    }

    return Document(
        id=doc_id,
        content=content,
        title=title,
        source=source,
        source_type="file",
        metadata=metadata,
    )


def parse_pdf(file_path: str) -> Document:
    """Parse PDF file into a Document.

    Args:
        file_path: Path to the PDF file

    Returns:
        Document object
    """
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        msg = "pdfminer.six is required for PDF parsing. Install with: pip install pdfminer.six"
        raise ImportError(msg)

    # Extract text from PDF
    content = extract_text(file_path)

    # Extract title from filename
    filename = os.path.basename(file_path)
    title = filename.rsplit('.', 1)[0]  # Remove extension

    doc_id = md5(f"{file_path}:{title}".encode()).hexdigest()

    metadata = {
        "file_type": "pdf",
    }

    return Document(
        id=doc_id,
        content=content,
        title=title,
        source=file_path,
        source_type="file",
        metadata=metadata,
    )


def chunk_document(doc: Document, chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """Split a document into smaller chunks.

    Args:
        doc: Document to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of Document objects representing chunks
    """
    content = doc.content

    # If document is small enough, return as-is
    if len(content) <= chunk_size:
        return [doc]

    chunks = []
    start = 0

    while start < len(content):
        # Calculate end position
        end = start + chunk_size

        # Find the last space to avoid cutting words
        if end < len(content):
            # Look for last space within chunk_size characters
            last_space = content.rfind(' ', start, end)
            if last_space > start:
                end = last_space

        # Extract chunk
        chunk_content = content[start:end].strip()

        if chunk_content:  # Only add non-empty chunks
            chunk_title = f"{doc.title} [chunk {len(chunks) + 1}]"
            chunk_id = md5(f"{doc.source}:{chunk_title}".encode()).hexdigest()

            chunk_doc = Document(
                id=chunk_id,
                content=chunk_content,
                title=chunk_title,
                source=doc.source,
                source_type=doc.source_type,
                metadata={**doc.metadata, "chunk_index": len(chunks)},
            )
            chunks.append(chunk_doc)

        # Move start position for next chunk (with overlap)
        start = end - overlap if end < len(content) else len(content)

    return chunks if chunks else [doc]


def extract_title_from_markdown(content: str, source: str) -> str:
    """Extract title from markdown content.

    Args:
        content: Markdown content
        source: Source path for fallback

    Returns:
        Title string
    """
    lines = content.split('\n')

    # Look for first heading
    for line in lines:
        if line.startswith('#'):
            # Remove markdown heading syntax
            title = line.lstrip('#').strip()
            if title:
                return title

    # Fallback to filename
    filename = os.path.basename(source)
    return filename.rsplit('.', 1)[0]


def extract_title_from_text(content: str, source: str) -> str:
    """Extract title from text content.

    Args:
        content: Text content
        source: Source path for fallback

    Returns:
        Title string
    """
    # Use first line if it's short enough to be a title
    lines = content.split('\n')
    if lines and len(lines[0]) < 100 and len(lines[0]) > 0:
        return lines[0].strip()

    # Fallback to filename
    filename = os.path.basename(source)
    return filename.rsplit('.', 1)[0]

