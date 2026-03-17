"""Tests for CodeEntity model and conversion."""
import pytest
from src.models.code_entity import CodeEntity, EntityType, code_entity_to_document
from src.models.document import Document


class TestCodeEntity:
    """Test CodeEntity dataclass."""

    def test_code_entity_creation(self):
        """Test CodeEntity dataclass creation."""
        entity = CodeEntity(
            name="my_function",
            entity_type=EntityType.FUNCTION,
            file_path="src/module.py",
            start_line=10,
            end_line=20,
            docstring='"""This is a function."""',
            source_code="def my_function():\n    pass",
            parent_name=None,
            decorators=["@decorator"],
            parameters=["x", "y"]
        )
        
        assert entity.name == "my_function"
        assert entity.entity_type == EntityType.FUNCTION
        assert entity.file_path == "src/module.py"
        assert entity.start_line == 10
        assert entity.end_line == 20
        assert entity.docstring == '"""This is a function."""'
        assert entity.source_code == "def my_function():\n    pass"
        assert entity.parent_name is None
        assert entity.decorators == ["@decorator"]
        assert entity.parameters == ["x", "y"]

    def test_code_entity_to_document(self):
        """Test conversion of CodeEntity to Document."""
        entity = CodeEntity(
            name="my_function",
            entity_type=EntityType.FUNCTION,
            file_path="src/module.py",
            start_line=10,
            end_line=20,
            docstring='"""This is a function."""',
            source_code="def my_function():\n    pass",
            parent_name=None,
            decorators=["@decorator"],
            parameters=["x", "y"]
        )
        
        doc = code_entity_to_document(entity)
        
        assert isinstance(doc, Document)
        assert doc.source_type == "code"
        assert "my_function" in doc.title
        assert doc.source == "src/module.py"
        # Content should include docstring and source code
        assert '"""This is a function."""' in doc.content
        assert "def my_function():\n    pass" in doc.content
        assert "src/module.py" in doc.metadata.get("full_path", "")

    def test_code_entity_to_document_no_docstring(self):
        """Test conversion to Document when there's no docstring."""
        entity = CodeEntity(
            name="my_function",
            entity_type=EntityType.FUNCTION,
            file_path="src/module.py",
            start_line=10,
            end_line=20,
            docstring=None,
            source_code="def my_function():\n    pass",
            parent_name=None
        )
        
        doc = code_entity_to_document(entity)
        
        assert isinstance(doc, Document)
        assert "def my_function():\n    pass" in doc.content
        assert doc.source_type == "code"

    def test_entity_type_enum_values(self):
        """Test EntityType enum has expected values."""
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.CLASS.value == "class"
        assert EntityType.METHOD.value == "method"
        assert EntityType.MODULE.value == "module"
        
        # Test that we can create from string
        assert EntityType("function") == EntityType.FUNCTION
        assert EntityType("class") == EntityType.CLASS
        assert EntityType("method") == EntityType.METHOD
        assert EntityType("module") == EntityType.MODULE

    def test_code_entity_default_decorators(self):
        """Test CodeEntity has default empty decorators list."""
        entity = CodeEntity(
            name="my_function",
            entity_type=EntityType.FUNCTION,
            file_path="src/module.py",
            start_line=10,
            end_line=20,
            docstring=None,
            source_code="def my_function():\n    pass",
            parent_name=None
        )
        
        assert entity.decorators == []
        assert entity.parameters == []

    def test_code_entity_method_with_parent(self):
        """Test CodeEntity for a method with parent class."""
        entity = CodeEntity(
            name="my_method",
            entity_type=EntityType.METHOD,
            file_path="src/module.py",
            start_line=15,
            end_line=20,
            docstring='"""Method docstring."""',
            source_code="def my_method(self):\n    pass",
            parent_name="MyClass",
            decorators=["@staticmethod"]
        )
        
        assert entity.parent_name == "MyClass"
        assert entity.entity_type == EntityType.METHOD
        
        doc = code_entity_to_document(entity)
        assert "MyClass" in doc.metadata.get("parent_name", "")
