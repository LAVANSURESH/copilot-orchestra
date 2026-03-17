"""Tests for CodeRepositorySource class."""
import pytest
from pathlib import Path
from src.ingestion.code_source import CodeRepositorySource
from src.models.code_entity import EntityType


class TestCodeRepositorySource:
    """Test CodeRepositorySource for loading Python files from repos."""

    def test_load_python_files_from_repo(self, tmp_path):
        """Test loading from a temp Python code directory."""
        # Create sample Python files
        py_file1 = tmp_path / "module1.py"
        py_file1.write_text("""
def function1():
    '''Function 1.'''
    pass
""")
        
        py_file2 = tmp_path / "module2.py"
        py_file2.write_text("""
class MyClass:
    '''My class.'''
    
    def method(self):
        pass
""")
        
        # Load from repo
        source = CodeRepositorySource(repo_paths=[str(tmp_path)])
        documents = source.load()
        
        # Should have multiple documents (functions, classes, modules, methods)
        assert len(documents) > 0
        assert all(doc.source_type == "code" for doc in documents)
        
        # Should have documents from both files
        sources = {doc.source for doc in documents}
        assert any("module1.py" in s for s in sources)
        assert any("module2.py" in s for s in sources)

    def test_excludes_test_directories(self, tmp_path):
        """Test that test dirs are excluded."""
        # Create Python files in test directory
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        
        test_file = test_dir / "test_module.py"
        test_file.write_text("""
def test_something():
    pass
""")
        
        # Create Python file in non-test directory
        src_file = tmp_path / "module.py"
        src_file.write_text("""
def my_function():
    pass
""")
        
        # Load from repo - should exclude test directory
        source = CodeRepositorySource(repo_paths=[str(tmp_path)])
        documents = source.load()
        
        # Should not have documents from test directory
        sources = {doc.source for doc in documents}
        assert not any("test" in s.lower() and "test_module" in s for s in sources)
        # But should have documents from src_file
        assert any("module.py" in s for s in sources)

    def test_get_entities_returns_code_entities(self, tmp_path):
        """Test getting raw entities."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
def my_function():
    '''Function doc.'''
    pass

class MyClass:
    '''Class doc.'''
    pass
""")
        
        source = CodeRepositorySource(repo_paths=[str(tmp_path)])
        entities = source.get_entities()
        
        assert len(entities) > 0
        
        # Should have different entity types
        entity_types = {e.entity_type for e in entities}
        assert EntityType.FUNCTION in entity_types or EntityType.CLASS in entity_types

    def test_loads_only_py_files(self, tmp_path):
        """Test that only .py files are loaded."""
        # Create various file types
        py_file = tmp_path / "module.py"
        py_file.write_text("""
def my_function():
    pass
""")
        
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Some text")
        
        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value"}')
        
        # Load from repo
        source = CodeRepositorySource(repo_paths=[str(tmp_path)])
        documents = source.load()
        
        # Should only have documents from Python files
        assert len(documents) > 0
        sources = {doc.source for doc in documents}
        assert any("module.py" in s for s in sources)
        assert not any("readme.txt" in s for s in sources)
        assert not any("data.json" in s for s in sources)

    def test_recursive_directory_loading(self, tmp_path):
        """Test recursive loading of directories."""
        # Create nested structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        src_file = src_dir / "module.py"
        src_file.write_text("""
def nested_function():
    pass
""")
        
        root_file = tmp_path / "root.py"
        root_file.write_text("""
def root_function():
    pass
""")
        
        # Load with recursive=True
        source = CodeRepositorySource(repo_paths=[str(tmp_path)], recursive=True)
        documents = source.load()
        
        # Should have documents from both root and nested files
        assert len(documents) > 0
        sources = {doc.source for doc in documents}
        assert any("root.py" in s for s in sources)
        assert any("module.py" in s for s in sources)

    def test_non_recursive_directory_loading(self, tmp_path):
        """Test non-recursive directory loading."""
        # Create nested structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        src_file = src_dir / "module.py"
        src_file.write_text("""
def nested_function():
    pass
""")
        
        root_file = tmp_path / "root.py"
        root_file.write_text("""
def root_function():
    pass
""")
        
        # Load with recursive=False
        source = CodeRepositorySource(repo_paths=[str(tmp_path)], recursive=False)
        documents = source.load()
        
        # Should only have documents from root level
        sources = {doc.source for doc in documents}
        assert any("root.py" in s for s in sources)
        # Nested file should not be included
        assert not any("module.py" in s for s in sources)

    def test_excludes_common_patterns(self, tmp_path):
        """Test that common exclude patterns work."""
        # Create excluded directories
        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        cache_file = pycache_dir / "module.py"
        cache_file.write_text("# Cached file")
        
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        venv_file = venv_dir / "module.py"
        venv_file.write_text("# Virtual env file")
        
        # Create included file
        src_file = tmp_path / "src.py"
        src_file.write_text("def func(): pass")
        
        # Load
        source = CodeRepositorySource(repo_paths=[str(tmp_path)])
        documents = source.load()
        
        # Should not include files from excluded directories
        sources = {doc.source for doc in documents}
        assert not any("__pycache__" in s for s in sources)
        assert not any("venv" in s for s in sources)
        # But should include the src.py
        assert any("src.py" in s for s in sources)

    def test_get_source_id(self, tmp_path):
        """Test getting source ID."""
        py_file = tmp_path / "module.py"
        py_file.write_text("def func(): pass")
        
        source = CodeRepositorySource(repo_paths=[str(tmp_path)])
        source_id = source.get_source_id()
        
        assert isinstance(source_id, str)
        assert len(source_id) > 0

    def test_custom_exclude_patterns(self, tmp_path):
        """Test custom exclude patterns."""
        # Create a custom directory to exclude
        custom_dir = tmp_path / "myexclude"  # Changed name to avoid collision with "custom" in path strings
        custom_dir.mkdir()
        custom_file = custom_dir / "module.py"
        custom_file.write_text("def func(): pass")
        
        # Create regular file
        src_file = tmp_path / "src.py"
        src_file.write_text("def func(): pass")
        
        # Load with custom exclude patterns
        source = CodeRepositorySource(
            repo_paths=[str(tmp_path)],
            exclude_patterns=["myexclude", "__pycache__"]
        )
        documents = source.load()
        
        # Should exclude custom directory
        sources = {doc.source for doc in documents}
        assert not any("myexclude" in s for s in sources)
        assert any("src.py" in s for s in sources)
