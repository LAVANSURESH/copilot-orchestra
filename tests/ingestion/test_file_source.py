"""Tests for file system document source."""
import pytest
import tempfile
import os
from pathlib import Path
from src.ingestion.file_source import FileSystemSource


class TestFileSystemSource:
    """Test file system document loading."""

    def test_load_markdown_files(self, tmp_path):
        """Test loading markdown files from temp directory."""
        # Create test markdown files
        md_file1 = tmp_path / "doc1.md"
        md_file1.write_text("# Document 1\n\nThis is content 1.")
        
        md_file2 = tmp_path / "doc2.md"
        md_file2.write_text("# Document 2\n\nThis is content 2.")
        
        # Load documents
        source = FileSystemSource(paths=[str(tmp_path)])
        documents = source.load()
        
        # Should load 2 documents
        assert len(documents) == 2
        assert all(doc.source_type == "file" for doc in documents)
        
        # Both should have content
        contents = {doc.content for doc in documents}
        assert "# Document 1\n\nThis is content 1." in contents
        assert "# Document 2\n\nThis is content 2." in contents

    def test_load_text_files(self, tmp_path):
        """Test loading text files."""
        # Create test text files
        txt_file1 = tmp_path / "doc1.txt"
        txt_file1.write_text("Plain text content 1.")
        
        txt_file2 = tmp_path / "doc2.txt"
        txt_file2.write_text("Plain text content 2.")
        
        # Load documents
        source = FileSystemSource(paths=[str(tmp_path)])
        documents = source.load()
        
        # Should load 2 documents
        assert len(documents) == 2
        assert all(doc.source_type == "file" for doc in documents)

    def test_recursive_loading(self, tmp_path):
        """Test recursive directory loading."""
        # Create nested directories with files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        md_file1 = tmp_path / "doc1.md"
        md_file1.write_text("# Root doc")
        
        md_file2 = subdir / "doc2.md"
        md_file2.write_text("# Subdir doc")
        
        # Load with recursive=True
        source = FileSystemSource(paths=[str(tmp_path)], recursive=True)
        documents = source.load()
        
        # Should load 2 documents (one from root, one from subdir)
        assert len(documents) == 2
        
        sources = {doc.source for doc in documents}
        assert str(md_file1) in sources
        assert str(md_file2) in sources

    def test_non_recursive_loading(self, tmp_path):
        """Test non-recursive directory loading."""
        # Create nested directories with files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        md_file1 = tmp_path / "doc1.md"
        md_file1.write_text("# Root doc")
        
        md_file2 = subdir / "doc2.md"
        md_file2.write_text("# Subdir doc")
        
        # Load with recursive=False
        source = FileSystemSource(paths=[str(tmp_path)], recursive=False)
        documents = source.load()
        
        # Should load only 1 document (from root)
        assert len(documents) == 1
        assert documents[0].source == str(md_file1)

    def test_skip_unsupported_files(self, tmp_path):
        """Test that unsupported file types are skipped."""
        # Create supported and unsupported files
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Doc")
        
        unsupported_file = tmp_path / "data.json"
        unsupported_file.write_text('{"key": "value"}')
        
        # Load documents
        source = FileSystemSource(paths=[str(tmp_path)])
        documents = source.load()
        
        # Should load only the markdown file
        assert len(documents) == 1
        assert documents[0].source == str(md_file)

    def test_custom_file_types(self, tmp_path):
        """Test filtering by custom file types."""
        # Create various file types
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Doc")
        
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("Text content")
        
        rst_file = tmp_path / "readme.rst"
        rst_file.write_text("RST content")
        
        # Load only markdown files
        source = FileSystemSource(paths=[str(tmp_path)], file_types=[".md"])
        documents = source.load()
        
        # Should load only markdown file
        assert len(documents) == 1
        assert documents[0].source == str(md_file)

    def test_get_source_id(self, tmp_path):
        """Test getting the source ID."""
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Doc")
        
        source = FileSystemSource(paths=[str(tmp_path)])
        source_id = source.get_source_id()
        
        # Should return a non-empty string
        assert isinstance(source_id, str)
        assert len(source_id) > 0
