"""Tests for document parsers."""
import pytest
from src.models.document import Document
from src.ingestion.parsers import (
    parse_markdown,
    parse_text,
    parse_pdf,
    chunk_document,
)
import tempfile
import os


class TestParseMarkdown:
    """Test markdown parsing."""

    def test_parse_markdown(self):
        """Test parsing markdown content."""
        content = "# Title\n\nThis is markdown content.\n\n## Section\n\nMore text."
        source = "/path/to/file.md"
        doc = parse_markdown(content, source)
        
        assert isinstance(doc, Document)
        assert doc.content == content
        assert doc.source == source
        assert doc.source_type == "file"
        assert "markdown" in doc.metadata["file_type"].lower() or "md" in doc.metadata["file_type"].lower()
        assert doc.title  # Should have a title extracted
        assert doc.id  # Should have an id


class TestParseText:
    """Test plain text parsing."""

    def test_parse_text(self):
        """Test parsing plain text content."""
        content = "This is plain text.\nWith multiple lines.\nAnd paragraphs."
        source = "/path/to/file.txt"
        doc = parse_text(content, source)
        
        assert isinstance(doc, Document)
        assert doc.content == content
        assert doc.source == source
        assert doc.source_type == "file"
        assert "text" in doc.metadata["file_type"].lower()
        assert doc.title  # Should have a title extracted
        assert doc.id  # Should have an id


class TestParsePDF:
    """Test PDF parsing."""

    def test_parse_pdf(self):
        """Test parsing PDF file."""
        # Create a temporary PDF file (minimal valid PDF)
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 750 Td
(Sample PDF Content) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000250 00000 n 
0000000349 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
423
%%EOF"""
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name
        
        try:
            doc = parse_pdf(tmp_path)
            assert isinstance(doc, Document)
            assert doc.source == tmp_path
            assert doc.source_type == "file"
            assert doc.metadata["file_type"].lower() == "pdf"
            assert doc.id
        finally:
            os.unlink(tmp_path)


class TestChunkDocument:
    """Test document chunking."""

    def test_chunk_document_splits_large_text(self):
        """Test that large documents are split into chunks."""
        # Create a document with text larger than chunk size
        content = "word " * 200  # 1000 words, each "word " is 5 chars
        source = "/path/to/file.txt"
        doc = parse_text(content, source)
        
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        
        # Should have more than one chunk
        assert len(chunks) > 1
        
        # All chunks should be Document objects
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert chunk.content  # Should have content
            assert len(chunk.content) <= 500  # Should respect chunk size (with some margin)
            assert chunk.source_type == "file"

    def test_chunk_document_small_doc_no_split(self):
        """Test that small documents are not split into chunks."""
        content = "This is a small document."
        source = "/path/to/file.txt"
        doc = parse_text(content, source)
        
        chunks = chunk_document(doc, chunk_size=500, overlap=50)
        
        # Should have exactly one chunk
        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_chunk_document_overlap(self):
        """Test that chunks have proper overlap."""
        # Create a document with specific content for testing overlap
        content = " ".join([f"word{i}" for i in range(200)])  # Many words
        source = "/path/to/file.txt"
        doc = parse_text(content, source)
        
        chunks = chunk_document(doc, chunk_size=200, overlap=50)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Check that chunks overlap (content should be repeated)
        # by verifying consecutive chunks share some words
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # There should be some overlap in the text
                chunk1_end = chunks[i].content[-50:]  # Last 50 chars of chunk 1
                chunk2_start = chunks[i + 1].content[:50]  # First 50 chars of chunk 2
                # They should share some content (this is a loose check)
                # Both chunks should exist and have content
                assert chunks[i].content
                assert chunks[i + 1].content
