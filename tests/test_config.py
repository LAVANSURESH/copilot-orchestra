"""Tests for configuration management."""

import pytest

from src.config.settings import Settings


class TestConfigLoadFromEnv:
    """Test loading configuration from environment variables."""

    def test_load_config_from_env(self, monkeypatch):
        """Test that config loads from environment variables."""
        # Set minimal required environment variables
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("WEAVIATE_PORT", "8080")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_AUTH_TYPE", "api_token")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "test@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "test-token")

        settings = Settings()
        assert settings.anthropic_api_key == "test-anthropic-key"
        assert settings.google_gemini_api_key == "test-gemini-key"
        assert settings.weaviate_host == "localhost"
        assert settings.weaviate_port == 8080
        assert settings.confluence_base_url == "https://test.atlassian.net"
        assert settings.confluence_auth_type == "api_token"
        assert settings.confluence_username == "test@example.com"
        assert settings.confluence_api_token == "test-token"

    def test_missing_required_config_raises_error(self, monkeypatch):
        """Test that missing required fields raise validation errors."""
        # Clear all relevant environment variables
        for key in [
            "ANTHROPIC_API_KEY",
            "GOOGLE_GEMINI_API_KEY",
            "WEAVIATE_HOST",
            "CONFLUENCE_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # Should raise validation error for missing required fields
        with pytest.raises(ValueError):
            Settings()

    def test_default_config_values(self, monkeypatch):
        """Test that default values are set correctly."""
        # Set only required fields
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")

        settings = Settings()

        # Test defaults
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.max_context_tokens == 2000
        assert settings.max_conversation_history == 10
        assert settings.fastapi_host == "0.0.0.0"
        assert settings.fastapi_port == 8000
        assert settings.weaviate_port == 8080
        assert settings.use_embedded_weaviate is False
        assert settings.weaviate_api_key is None

    def test_confluence_api_token_auth_config(self, monkeypatch):
        """Test Confluence API token authentication configuration."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_AUTH_TYPE", "api_token")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "test@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "test-api-token")

        settings = Settings()

        assert settings.confluence_auth_type == "api_token"
        assert settings.confluence_username == "test@example.com"
        assert settings.confluence_api_token == "test-api-token"
        assert settings.confluence_oauth_client_id is None
        assert settings.confluence_oauth_client_secret is None
        assert settings.confluence_oauth_token_url is None

    def test_confluence_oauth_config(self, monkeypatch):
        """Test Confluence OAuth authentication configuration."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_AUTH_TYPE", "oauth")
        monkeypatch.setenv("CONFLUENCE_OAUTH_CLIENT_ID", "client-id")
        monkeypatch.setenv("CONFLUENCE_OAUTH_CLIENT_SECRET", "client-secret")
        monkeypatch.setenv("CONFLUENCE_OAUTH_TOKEN_URL", "https://test.atlassian.net/oauth/token")

        settings = Settings()

        assert settings.confluence_auth_type == "oauth"
        assert settings.confluence_oauth_client_id == "client-id"
        assert settings.confluence_oauth_client_secret == "client-secret"
        assert settings.confluence_oauth_token_url == "https://test.atlassian.net/oauth/token"
        assert settings.confluence_username is None
        assert settings.confluence_api_token is None

    def test_optional_config_fields(self, monkeypatch):
        """Test that optional config fields can be None."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")

        settings = Settings()

        # These should be optional and default to None
        assert settings.document_paths is None or isinstance(settings.document_paths, (list, str))
        assert settings.code_repo_paths is None or isinstance(settings.code_repo_paths, (list, str))
        assert settings.weaviate_api_key is None

    def test_weaviate_embedded_config(self, monkeypatch):
        """Test Weaviate embedded configuration."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")
        monkeypatch.setenv("USE_EMBEDDED_WEAVIATE", "true")
        monkeypatch.setenv("WEAVIATE_API_KEY", "embedded-key")

        settings = Settings()

        assert settings.use_embedded_weaviate is True
        assert settings.weaviate_api_key == "embedded-key"

    def test_custom_max_tokens_config(self, monkeypatch):
        """Test custom max tokens configuration."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")
        monkeypatch.setenv("MAX_CONTEXT_TOKENS", "4000")
        monkeypatch.setenv("MAX_CONVERSATION_HISTORY", "20")

        settings = Settings()

        assert settings.max_context_tokens == 4000
        assert settings.max_conversation_history == 20

    def test_fastapi_server_config(self, monkeypatch):
        """Test FastAPI server configuration."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("WEAVIATE_HOST", "localhost")
        monkeypatch.setenv("CONFLUENCE_BASE_URL", "https://test.atlassian.net")
        monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")
        monkeypatch.setenv("FASTAPI_PORT", "9000")

        settings = Settings()

        assert settings.fastapi_host == "127.0.0.1"
        assert settings.fastapi_port == 9000
