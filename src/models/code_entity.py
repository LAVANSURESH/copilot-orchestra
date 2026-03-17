"""Code entity model for representing Python code elements."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from src.models.document import Document


class EntityType(str, Enum):
    """Enum for types of code entities."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"


@dataclass
class CodeEntity:
    """Represents a Python code entity (function, class, method, or module)."""

    name: str  # Name of the entity
    entity_type: EntityType  # Type of entity
    file_path: str  # Path to the file
    start_line: int  # Starting line number
    end_line: int  # Ending line number
    docstring: Optional[str]  # Extracted docstring
    source_code: str  # Raw source code
    parent_name: Optional[str] = None  # For methods, the class name
    decorators: list[str] = field(default_factory=list)  # List of decorator names
    parameters: list[str] = field(default_factory=list)  # For functions/methods


def code_entity_to_document(entity: CodeEntity) -> Document:
    """Convert a CodeEntity to a Document for vector storage.

    Args:
        entity: CodeEntity to convert

    Returns:
        Document object with code content and metadata
    """
    # Build content from docstring and source code
    if entity.docstring:
        content = f"{entity.docstring}\n{entity.source_code}"
    else:
        content = entity.source_code

    # Create title
    if entity.parent_name:
        title = f"{entity.parent_name}.{entity.name}"
    else:
        title = entity.name

    # Build source reference
    source_ref = entity.file_path

    # Build metadata
    metadata = {
        "entity_type": entity.entity_type.value,
        "file_path": entity.file_path,
        "start_line": entity.start_line,
        "end_line": entity.end_line,
        "full_path": f"{entity.file_path}::{entity.entity_type.value}::{title}",
    }

    if entity.parent_name:
        metadata["parent_name"] = entity.parent_name

    if entity.decorators:
        metadata["decorators"] = entity.decorators

    if entity.parameters:
        metadata["parameters"] = entity.parameters

    # Create Document
    doc = Document(
        id="",  # Will be auto-generated in __post_init__
        content=content,
        title=title,
        source=source_ref,
        source_type="code",
        metadata=metadata,
    )

    return doc
