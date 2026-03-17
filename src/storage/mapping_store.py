"""MappingStore for persisting semantic relationships between code and documentation."""

import json
import os
from pathlib import Path


class MappingStore:
    """Persists and retrieves semantic mappings between code and documentation."""

    def __init__(self, storage_path: str = ".chatbot_cache/mappings.json"):
        """Initialize the mapping store.

        Args:
            storage_path: Path to store mappings JSON file. Defaults to .chatbot_cache/mappings.json
        """
        self.storage_path = storage_path
        self._mappings = {}
        self._load_on_init()

    def _load_on_init(self):
        """Load mappings on initialization if file exists."""
        if os.path.exists(self.storage_path):
            self._mappings = self.load_mappings()

    def save_mappings(self, mappings: dict[str, list[str]]) -> None:
        """Save mappings to JSON file.

        Args:
            mappings: Dictionary of mappings to save.
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Write to file
        with open(self.storage_path, "w") as f:
            json.dump(mappings, f, indent=2)

        # Update in-memory cache
        self._mappings = mappings

    def load_mappings(self) -> dict[str, list[str]]:
        """Load mappings from JSON file.

        Returns:
            Dictionary of mappings, or empty dict if file doesn't exist.
        """
        if not os.path.exists(self.storage_path):
            return {}

        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def get_related(self, entity_id: str) -> list[str]:
        """Get IDs related to a given entity.

        Args:
            entity_id: The entity ID to look up.

        Returns:
            List of related entity IDs, or empty list if not found.
        """
        # Load fresh from file to ensure consistency
        current = self.load_mappings()
        return current.get(entity_id, [])

    def add_mapping(self, source_id: str, target_id: str) -> None:
        """Add a bidirectional mapping between two entities.

        Args:
            source_id: First entity ID.
            target_id: Second entity ID.
        """
        # Load current mappings
        current = self.load_mappings()

        # Add forward mapping
        if source_id not in current:
            current[source_id] = []
        if target_id not in current[source_id]:
            current[source_id].append(target_id)

        # Add reverse mapping
        if target_id not in current:
            current[target_id] = []
        if source_id not in current[target_id]:
            current[target_id].append(source_id)

        # Save updated mappings
        self.save_mappings(current)
