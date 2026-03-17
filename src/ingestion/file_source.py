"""File system document source."""
from pathlib import Path
from typing import List, Optional
from hashlib import md5
from src.ingestion.base import DocumentSource
from src.models.document import Document
from src.ingestion.parsers import parse_markdown, parse_text, parse_pdf


class FileSystemSource(DocumentSource):
    """Load documents from local file system."""

    DEFAULT_FILE_TYPES = [".md", ".txt", ".pdf", ".rst"]

    def __init__(
        self,
        paths: List[str],
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
    ):
        """Initialize file system source.

        Args:
            paths: List of directory or file paths to load from
            recursive: Whether to recursively search subdirectories
            file_types: List of file extensions to load (default: ['.md', '.txt', '.pdf', '.rst'])
        """
        self.paths = paths
        self.recursive = recursive
        self.file_types = file_types or self.DEFAULT_FILE_TYPES

    def load(self) -> List[Document]:
        """Load all supported documents from configured paths.

        Returns:
            List of Document objects
        """
        documents = []

        for path_str in self.paths:
            path = Path(path_str)

            if path.is_file():
                # Single file
                doc = self._load_file(path)
                if doc:
                    documents.append(doc)
            elif path.is_dir():
                # Directory
                files = self._find_files(path)
                for file_path in files:
                    doc = self._load_file(file_path)
                    if doc:
                        documents.append(doc)

        return documents

    def _find_files(self, directory: Path) -> List[Path]:
        """Find all supported files in directory.

        Args:
            directory: Directory to search

        Returns:
            List of file paths
        """
        files = []

        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.file_types:
                files.append(file_path)

        return files

    def _load_file(self, file_path: Path) -> Optional[Document]:
        """Load a single file.

        Args:
            file_path: Path to file

        Returns:
            Document object or None if file couldn't be loaded
        """
        file_ext = file_path.suffix.lower()

        try:
            if file_ext == ".md":
                content = file_path.read_text(encoding='utf-8')
                return parse_markdown(content, str(file_path))

            elif file_ext == ".txt":
                content = file_path.read_text(encoding='utf-8')
                return parse_text(content, str(file_path))

            elif file_ext == ".pdf":
                return parse_pdf(str(file_path))

            elif file_ext == ".rst":
                # Treat RST as text for now
                content = file_path.read_text(encoding='utf-8')
                return parse_text(content, str(file_path))

        except Exception as e:
            # Skip files that can't be read
            print(f"Warning: Could not load {file_path}: {e}")
            return None

        return None

    def get_source_id(self) -> str:
        """Get unique identifier for this source.

        Returns:
            String identifier
        """
        paths_str = "|".join(sorted(self.paths))
        return md5(paths_str.encode()).hexdigest()

