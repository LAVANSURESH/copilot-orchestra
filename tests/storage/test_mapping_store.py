"""Tests for mapping store."""

import pytest
import json
import os
import tempfile
import shutil
from src.storage.mapping_store import MappingStore


class TestMappingStore:
    """Test suite for MappingStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a mapping store with temp directory."""
        store_path = os.path.join(temp_dir, "mappings.json")
        return MappingStore(storage_path=store_path)

    @pytest.fixture
    def sample_mappings(self):
        """Create sample mappings."""
        return {
            "code:authenticate": ["doc:auth_guide", "doc:login_api"],
            "doc:auth_guide": ["code:authenticate", "code:validate"],
            "code:validate": ["doc:auth_guide"],
            "doc:login_api": ["code:authenticate"],
        }

    def test_save_and_load_mappings(self, store, sample_mappings):
        """Test saving and loading mappings."""
        # Save mappings
        store.save_mappings(sample_mappings)

        # Load mappings
        loaded = store.load_mappings()

        # Verify
        assert loaded == sample_mappings
        assert loaded["code:authenticate"] == ["doc:auth_guide", "doc:login_api"]

    def test_get_related_returns_correct_ids(self, store, sample_mappings):
        """Test retrieving related IDs for an entity."""
        # Save mappings
        store.save_mappings(sample_mappings)

        # Get related
        related = store.get_related("code:authenticate")

        # Verify
        assert related == ["doc:auth_guide", "doc:login_api"]
        assert len(related) == 2

    def test_add_mapping_creates_bidirectional(self, store):
        """Test adding a bidirectional mapping."""
        # Add mapping
        store.add_mapping("code:func1", "doc:doc1")

        # Verify both directions exist
        assert "code:func1" in store.get_related("doc:doc1")
        assert "doc:doc1" in store.get_related("code:func1")

    def test_load_empty_store(self, store):
        """Test loading from non-existent file."""
        # Load without saving
        loaded = store.load_mappings()

        # Verify - should return empty dict
        assert loaded == {}

    def test_get_related_nonexistent_entity(self, store, sample_mappings):
        """Test getting related for entity that doesn't exist."""
        # Save mappings
        store.save_mappings(sample_mappings)

        # Get related for non-existent entity
        related = store.get_related("nonexistent:entity")

        # Verify - should return empty list
        assert related == []

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates necessary directories."""
        nested_path = os.path.join(temp_dir, "subdir", "nested", "mappings.json")
        store = MappingStore(storage_path=nested_path)

        # Save mappings
        store.save_mappings({"code:test": ["doc:test"]})

        # Verify file exists
        assert os.path.exists(nested_path)

    def test_save_overwrites_existing(self, store, sample_mappings):
        """Test that save overwrites existing mappings."""
        # Save initial mappings
        store.save_mappings(sample_mappings)

        # Save new mappings
        new_mappings = {"code:new": ["doc:new"]}
        store.save_mappings(new_mappings)

        # Load and verify
        loaded = store.load_mappings()
        assert loaded == new_mappings
        assert "code:authenticate" not in loaded

    def test_add_mapping_to_empty_store(self, store):
        """Test adding mapping to empty store."""
        # Add mapping
        store.add_mapping("code:test", "doc:test")

        # Load and verify
        loaded = store.load_mappings()
        assert "code:test" in loaded
        assert "doc:test" in loaded

    def test_add_multiple_mappings(self, store):
        """Test adding multiple mappings."""
        # Add multiple
        store.add_mapping("code:func1", "doc:doc1")
        store.add_mapping("code:func2", "doc:doc1")
        store.add_mapping("code:func1", "doc:doc2")

        # Verify
        assert "doc:doc1" in store.get_related("code:func1")
        assert "doc:doc2" in store.get_related("code:func1")
        assert "code:func1" in store.get_related("doc:doc1")
        assert "code:func2" in store.get_related("doc:doc1")

    def test_add_duplicate_mapping(self, store):
        """Test adding duplicate mapping doesn't create duplicates."""
        # Add mapping twice
        store.add_mapping("code:func", "doc:doc")
        store.add_mapping("code:func", "doc:doc")

        # Verify no duplicates
        related = store.get_related("code:func")
        assert related.count("doc:doc") == 1

    def test_mappings_file_format(self, store, sample_mappings):
        """Test that mappings are saved in valid JSON format."""
        # Save mappings
        store.save_mappings(sample_mappings)

        # Read file directly
        with open(store.storage_path, "r") as f:
            content = json.load(f)

        # Verify it's valid JSON
        assert isinstance(content, dict)
        assert content == sample_mappings

    def test_get_related_preserves_order(self, store):
        """Test that get_related preserves order of additions."""
        # Add mappings in specific order
        store.add_mapping("code:func", "doc:doc1")
        store.add_mapping("code:func", "doc:doc2")
        store.add_mapping("code:func", "doc:doc3")

        # Get related
        related = store.get_related("code:func")

        # Verify order is preserved
        assert related[0] == "doc:doc1"
        assert related[1] == "doc:doc2"
        assert related[2] == "doc:doc3"

    def test_default_storage_path(self):
        """Test default storage path."""
        store = MappingStore()

        # Verify default path
        assert ".chatbot_cache" in store.storage_path
        assert "mappings.json" in store.storage_path

    def test_custom_storage_path(self, temp_dir):
        """Test custom storage path."""
        custom_path = os.path.join(temp_dir, "custom_mappings.json")
        store = MappingStore(storage_path=custom_path)

        # Save and verify
        store.save_mappings({"test": ["data"]})
        assert os.path.exists(custom_path)

    def test_empty_mappings_save_load(self, store):
        """Test saving and loading empty mappings."""
        # Save empty
        store.save_mappings({})

        # Load
        loaded = store.load_mappings()

        # Verify
        assert loaded == {}

    def test_complex_mapping_structure(self, store):
        """Test complex mapping with many relationships."""
        complex_mappings = {
            "code:auth.authenticate": [
                "doc:auth_guide",
                "doc:login_api",
                "doc:security_best_practices",
                "code:auth.validate",
            ],
            "code:auth.validate": ["doc:auth_guide", "doc:validation_rules"],
            "doc:auth_guide": [
                "code:auth.authenticate",
                "code:auth.validate",
                "code:auth.hash_password",
            ],
        }

        # Save and load
        store.save_mappings(complex_mappings)
        loaded = store.load_mappings()

        # Verify complex structure
        assert loaded == complex_mappings
        assert len(loaded["code:auth.authenticate"]) == 4
