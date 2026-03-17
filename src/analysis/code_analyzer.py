"""Code analyzer for extracting code entities from Python files."""

import ast
from pathlib import Path
from typing import List, Optional
from src.models.code_entity import CodeEntity, EntityType


class CodeAnalyzer:
    """Analyzes Python files to extract code entities."""

    def analyze_file(self, file_path: str) -> List[CodeEntity]:
        """Parse a Python file and extract all code entities.

        Args:
            file_path: Path to the Python file

        Returns:
            List of CodeEntity objects extracted from the file
        """
        path = Path(file_path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error reading file {file_path}: {e}")
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return []

        entities = []

        # Add module entity
        module_entity = self._create_module_entity(tree, file_path, source)
        entities.append(module_entity)

        # Extract functions and classes
        entities.extend(self._extract_functions(tree, file_path, source))
        entities.extend(self._extract_classes(tree, file_path, source))

        return entities

    def _create_module_entity(self, tree: ast.AST, file_path: str, source: str) -> CodeEntity:
        """Create a module entity for the file.

        Args:
            tree: AST tree of the module
            file_path: Path to the file
            source: Source code

        Returns:
            CodeEntity representing the module
        """
        # Get module docstring
        docstring = ast.get_docstring(tree)

        # Get module name from file path
        module_name = Path(file_path).stem

        return CodeEntity(
            name=module_name,
            entity_type=EntityType.MODULE,
            file_path=file_path,
            start_line=1,
            end_line=len(source.splitlines()),
            docstring=docstring,
            source_code=source,
            parent_name=None,
        )

    def _extract_functions(self, tree: ast.AST, file_path: str, source: str) -> List[CodeEntity]:
        """Extract all top-level functions from AST.

        Args:
            tree: AST tree
            file_path: Path to file
            source: Source code

        Returns:
            List of CodeEntity objects for functions
        """
        entities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                # Only top-level functions (col_offset == 0)
                # Skip methods (they're handled in _extract_classes)
                # Check if this is inside a class by seeing if any ClassDef contains it
                if self._is_function_in_class(tree, node):
                    continue

                entity = self._create_function_entity(node, file_path, source, parent_name=None)
                entities.append(entity)

        return entities

    def _extract_classes(self, tree: ast.AST, file_path: str, source: str) -> List[CodeEntity]:
        """Extract classes and their methods from AST.

        Args:
            tree: AST tree
            file_path: Path to file
            source: Source code

        Returns:
            List of CodeEntity objects for classes and their methods
        """
        entities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.col_offset == 0:
                # Top-level class
                class_entity = self._create_class_entity(node, file_path, source)
                entities.append(class_entity)

                # Extract methods from this class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_entity = self._create_method_entity(
                            item, file_path, source, class_name=node.name
                        )
                        entities.append(method_entity)

        return entities

    def _create_function_entity(
        self, node: ast.FunctionDef, file_path: str, source: str, parent_name: Optional[str] = None
    ) -> CodeEntity:
        """Create a CodeEntity for a function.

        Args:
            node: FunctionDef AST node
            file_path: Path to file
            source: Source code
            parent_name: Name of parent class if this is a method

        Returns:
            CodeEntity for the function
        """
        docstring = ast.get_docstring(node)
        source_code = self._get_source_segment(source, node)
        decorators = self._get_decorators(node)
        parameters = self._get_parameters(node)

        return CodeEntity(
            name=node.name,
            entity_type=EntityType.FUNCTION,
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            source_code=source_code,
            parent_name=parent_name,
            decorators=decorators,
            parameters=parameters,
        )

    def _create_method_entity(
        self, node: ast.FunctionDef, file_path: str, source: str, class_name: str
    ) -> CodeEntity:
        """Create a CodeEntity for a method.

        Args:
            node: FunctionDef AST node
            file_path: Path to file
            source: Source code
            class_name: Name of the class

        Returns:
            CodeEntity for the method
        """
        docstring = ast.get_docstring(node)
        source_code = self._get_source_segment(source, node)
        decorators = self._get_decorators(node)
        parameters = self._get_parameters(node)

        return CodeEntity(
            name=node.name,
            entity_type=EntityType.METHOD,
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            source_code=source_code,
            parent_name=class_name,
            decorators=decorators,
            parameters=parameters,
        )

    def _create_class_entity(self, node: ast.ClassDef, file_path: str, source: str) -> CodeEntity:
        """Create a CodeEntity for a class.

        Args:
            node: ClassDef AST node
            file_path: Path to file
            source: Source code

        Returns:
            CodeEntity for the class
        """
        docstring = ast.get_docstring(node)
        source_code = self._get_source_segment(source, node)
        decorators = self._get_decorators(node)

        return CodeEntity(
            name=node.name,
            entity_type=EntityType.CLASS,
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            source_code=source_code,
            parent_name=None,
            decorators=decorators,
        )

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Get the source code for a specific AST node.

        Args:
            source: Full source code
            node: AST node

        Returns:
            Source code for the node
        """
        try:
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                return segment
        except Exception:
            pass

        # Fallback: extract lines manually
        lines = source.splitlines(keepends=True)
        start_idx = node.lineno - 1
        end_idx = node.end_lineno or node.lineno

        if 0 <= start_idx < len(lines) and 0 <= end_idx <= len(lines):
            return "".join(lines[start_idx:end_idx])

        return ""

    def _get_decorators(self, node: ast.FunctionDef | ast.ClassDef) -> List[str]:
        """Extract decorator names from a function or class node.

        Args:
            node: FunctionDef or ClassDef AST node

        Returns:
            List of decorator names
        """
        decorators = []

        if not hasattr(node, "decorator_list"):
            return decorators

        for decorator in node.decorator_list:
            try:
                decorator_str = ast.unparse(decorator)
                decorators.append(decorator_str)
            except Exception:
                # Fallback for complex decorators
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(ast.unparse(decorator))

        return decorators

    def _get_parameters(self, node: ast.FunctionDef) -> List[str]:
        """Extract parameter names from a function.

        Args:
            node: FunctionDef AST node

        Returns:
            List of parameter names
        """
        if not hasattr(node, "args"):
            return []

        parameters = []

        # Regular arguments
        for arg in node.args.args:
            parameters.append(arg.arg)

        # Positional-only arguments (Python 3.8+)
        if hasattr(node.args, "posonlyargs"):
            for arg in node.args.posonlyargs:
                parameters.append(arg.arg)

        # Keyword-only arguments
        if hasattr(node.args, "kwonlyargs"):
            for arg in node.args.kwonlyargs:
                parameters.append(arg.arg)

        # *args
        if node.args.vararg:
            parameters.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            parameters.append(f"**{node.args.kwarg.arg}")

        return parameters

    def _is_function_in_class(self, tree: ast.AST, func_node: ast.FunctionDef) -> bool:
        """Check if a function is inside a class.

        Args:
            tree: AST tree
            func_node: FunctionDef node to check

        Returns:
            True if function is inside a class
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if item is func_node:
                        return True

        return False
