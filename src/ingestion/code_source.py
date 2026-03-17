"""Code repository document source."""

from pathlib import Path
from typing import List, Optional
from hashlib import md5
from src.ingestion.base import DocumentSource
from src.models.document import Document
from src.models.code_entity import CodeEntity
from src.analysis.code_analyzer import CodeAnalyzer


class CodeRepositorySource(DocumentSource):
    """Load code entities from Python repositories."""

    DEFAULT_EXCLUDE_PATTERNS = ["__pycache__", ".git", "node_modules", "venv", ".env", "tests"]

    def __init__(
        self,
        repo_paths: List[str],
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize code repository source.

        Args:
            repo_paths: List of repository paths to analyze
            recursive: Whether to recursively search subdirectories
            exclude_patterns: List of directory patterns to exclude
        """
        self.repo_paths = repo_paths
        self.recursive = recursive
        self.exclude_patterns = exclude_patterns or self.DEFAULT_EXCLUDE_PATTERNS
        self.analyzer = CodeAnalyzer()

    def load(self) -> List[Document]:
        """Load all Python files and convert to Documents.

        Returns:
            List of Document objects
        """
        documents = []
        entities = self.get_entities()

        for entity in entities:
            from src.models.code_entity import code_entity_to_document

            doc = code_entity_to_document(entity)
            documents.append(doc)

        return documents

    def get_entities(self) -> List[CodeEntity]:
        """Get raw CodeEntity objects.

        Returns:
            List of CodeEntity objects
        """
        entities = []

        for repo_path in self.repo_paths:
            path = Path(repo_path)

            if path.is_file():
                # Single file
                if path.suffix == ".py" and self._should_include(path):
                    entities.extend(self.analyzer.analyze_file(str(path)))
            elif path.is_dir():
                # Directory
                py_files = self._find_python_files(path)
                for py_file in py_files:
                    entities.extend(self.analyzer.analyze_file(str(py_file)))

        return entities

    def _find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in directory.

        Args:
            directory: Directory to search

        Returns:
            List of Python file paths
        """
        files = []

        if self.recursive:
            pattern = "**/*.py"
        else:
            pattern = "*.py"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and self._should_include(file_path):
                files.append(file_path)

        return files

    def _should_include(self, file_path: Path) -> bool:
        """Check if a file should be included based on exclude patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file should be included
        """
        # Check each path component (directory name) against exclude patterns
        # We match patterns as exact directory names (not substrings)
        for part in file_path.parts:
            # Check if the part matches any exclude pattern exactly
            if part in self.exclude_patterns:
                return False

        return True

    def get_source_id(self) -> str:
        """Get unique identifier for this source.

        Returns:
            String identifier
        """
        paths_str = "|".join(sorted(self.repo_paths))
        return md5(paths_str.encode()).hexdigest()
