"""Tests for CodeAnalyzer class."""
import pytest
from pathlib import Path
from src.analysis.code_analyzer import CodeAnalyzer
from src.models.code_entity import EntityType


class TestCodeAnalyzer:
    """Test CodeAnalyzer for extracting code entities."""

    def test_extract_functions_from_python(self, tmp_path):
        """Test function extraction from a Python file."""
        # Create a test Python file
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
def hello():
    '''Say hello.'''
    print("Hello")

def goodbye(name):
    '''Say goodbye to someone.'''
    return f"Goodbye {name}"
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        # Should have at least 2 functions + 1 module
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(functions) >= 2
        
        func_names = {f.name for f in functions}
        assert "hello" in func_names
        assert "goodbye" in func_names
        
        # Check function details
        hello = next(f for f in functions if f.name == "hello")
        assert hello.docstring is not None
        assert "Say hello" in hello.docstring
        assert hello.file_path == str(py_file)

    def test_extract_classes_from_python(self, tmp_path):
        """Test class extraction from a Python file."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
class Animal:
    '''Base animal class.'''
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        '''Make the animal speak.'''
        pass

class Dog(Animal):
    '''A dog is an animal.'''
    
    def speak(self):
        return "Woof"
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        # Should have classes
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) >= 2
        
        class_names = {c.name for c in classes}
        assert "Animal" in class_names
        assert "Dog" in class_names
        
        # Check class details
        animal = next(c for c in classes if c.name == "Animal")
        assert animal.docstring is not None
        assert "animal" in animal.docstring.lower()

    def test_extract_methods_from_class(self, tmp_path):
        """Test method extraction from a class."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
class Calculator:
    '''Simple calculator.'''
    
    def add(self, a, b):
        '''Add two numbers.'''
        return a + b
    
    def subtract(self, a, b):
        '''Subtract two numbers.'''
        return a - b
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        # Should have methods
        methods = [e for e in entities if e.entity_type == EntityType.METHOD]
        assert len(methods) >= 2
        
        method_names = {m.name for m in methods}
        assert "add" in method_names
        assert "subtract" in method_names
        
        # Check method parent
        add_method = next(m for m in methods if m.name == "add")
        assert add_method.parent_name == "Calculator"
        assert "Add two" in add_method.docstring

    def test_extract_docstrings(self, tmp_path):
        """Test docstring extraction."""
        py_file = tmp_path / "sample.py"
        py_file.write_text('''
def function_with_docstring():
    """This is a docstring."""
    pass

def function_without_docstring():
    pass
''')
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        
        # One should have docstring
        with_doc = next((f for f in functions if f.name == "function_with_docstring"), None)
        assert with_doc is not None
        assert with_doc.docstring is not None
        assert "docstring" in with_doc.docstring.lower()
        
        # One should not have docstring
        without_doc = next((f for f in functions if f.name == "function_without_docstring"), None)
        assert without_doc is not None
        assert without_doc.docstring is None

    def test_extract_decorators(self, tmp_path):
        """Test decorator extraction."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
@property
def name(self):
    return self._name

@staticmethod
def static_method():
    pass

@decorator1
@decorator2
def decorated_function():
    pass
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        
        # Find decorated function
        decorated = next((f for f in functions if f.name == "decorated_function"), None)
        assert decorated is not None
        assert len(decorated.decorators) >= 2
        # Decorators should contain the decorator names
        assert any("decorator1" in d or "decorator2" in d for d in decorated.decorators)

    def test_extract_function_parameters(self, tmp_path):
        """Test function parameter extraction."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
def no_params():
    pass

def with_params(x, y, z):
    pass

def with_defaults(a, b=10, c="default"):
    pass
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        
        no_params = next(f for f in functions if f.name == "no_params")
        assert no_params.parameters == []
        
        with_params = next(f for f in functions if f.name == "with_params")
        assert "x" in with_params.parameters
        assert "y" in with_params.parameters
        assert "z" in with_params.parameters
        
        with_defaults = next(f for f in functions if f.name == "with_defaults")
        assert "a" in with_defaults.parameters
        assert "b" in with_defaults.parameters

    def test_analyze_module_entity(self, tmp_path):
        """Test that the file itself is captured as a module entity."""
        py_file = tmp_path / "sample.py"
        py_file.write_text('''"""Module docstring."""
def sample_function():
    pass
''')
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        # Should have a module entity
        modules = [e for e in entities if e.entity_type == EntityType.MODULE]
        assert len(modules) == 1
        
        module = modules[0]
        assert module.name == "sample"  # or the module name
        assert module.file_path == str(py_file)
        assert module.docstring is not None
        assert "Module docstring" in module.docstring

    def test_line_numbers(self, tmp_path):
        """Test that start and end line numbers are captured."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
def function_one():
    pass

def function_two():
    x = 1
    y = 2
    return x + y
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        
        for func in functions:
            assert func.start_line > 0
            assert func.end_line > func.start_line

    def test_source_code_extraction(self, tmp_path):
        """Test that source code is properly extracted."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
def my_function():
    '''Docstring.'''
    return 42
""")
        
        analyzer = CodeAnalyzer()
        entities = analyzer.analyze_file(str(py_file))
        
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        my_func = next(f for f in functions if f.name == "my_function")
        
        assert my_func.source_code is not None
        assert "def my_function" in my_func.source_code
        assert "return 42" in my_func.source_code
