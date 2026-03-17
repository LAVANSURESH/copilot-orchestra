"""Configuration management for Enterprise Chatbot Framework using pydantic-settings."""

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main configuration class for the Enterprise Chatbot Framework.

    This class manages all configuration settings using pydantic-settings v2.
    It supports loading configuration from environment variables and .env files.

    Attributes:
        anthropic_api_key: API key for Anthropic Claude (required)
        google_gemini_api_key: API key for Google Gemini (required)
        weaviate_host: Weaviate server host (required)
        weaviate_port: Weaviate server port (optional, default: 8080)
        weaviate_api_key: API key for Weaviate (optional, for hosted Weaviate)
        use_embedded_weaviate: Use embedded Weaviate instead of server (optional, default: False)
        confluence_base_url: Confluence base URL (required)
        confluence_auth_type: Confluence auth type - "api_token" or "oauth" (required)
        confluence_username: Confluence username/email (optional, for API token auth)
        confluence_api_token: Confluence API token (optional, for API token auth)
        confluence_oauth_client_id: OAuth client ID (optional, for OAuth)
        confluence_oauth_client_secret: OAuth client secret (optional, for OAuth)
        confluence_oauth_token_url: OAuth token URL (optional, for OAuth)
        document_paths: Comma-separated document paths (optional)
        code_repo_paths: Comma-separated code repository paths (optional)
        embedding_model: HuggingFace model ID for embeddings (optional)
        max_context_tokens: Maximum context tokens (optional, default: 2000)
        max_conversation_history: Maximum conversation history messages (optional, default: 10)
        fastapi_host: FastAPI server host (optional, default: 0.0.0.0)
        fastapi_port: FastAPI server port (optional, default: 8000)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",
    )

    # ========================================================================
    # LLM Provider Configuration
    # ========================================================================

    anthropic_api_key: str = Field(
        ...,
        description="Anthropic Claude API key (required)",
    )

    google_gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key (required)",
    )

    # ========================================================================
    # Weaviate Configuration
    # ========================================================================

    weaviate_host: str = Field(
        ...,
        description="Weaviate server host (required)",
    )

    weaviate_port: int = Field(
        default=8080,
        description="Weaviate server port (default: 8080)",
    )

    weaviate_api_key: Optional[str] = Field(
        default=None,
        description="Weaviate API key for hosted instances (optional)",
    )

    use_embedded_weaviate: bool = Field(
        default=False,
        description="Use embedded Weaviate instead of server (default: False)",
    )

    # ========================================================================
    # Confluence Configuration
    # ========================================================================

    confluence_base_url: str = Field(
        ...,
        description="Confluence base URL (required)",
    )

    confluence_auth_type: str = Field(
        default="api_token",
        description="Confluence auth type: 'api_token' or 'oauth' (required)",
    )

    # API Token Authentication
    confluence_username: Optional[str] = Field(
        default=None,
        description="Confluence username/email for API token auth (optional)",
    )

    confluence_api_token: Optional[str] = Field(
        default=None,
        description="Confluence API token for API token auth (optional)",
    )

    # OAuth Authentication
    confluence_oauth_client_id: Optional[str] = Field(
        default=None,
        description="Confluence OAuth client ID (optional)",
    )

    confluence_oauth_client_secret: Optional[str] = Field(
        default=None,
        description="Confluence OAuth client secret (optional)",
    )

    confluence_oauth_token_url: Optional[str] = Field(
        default=None,
        description="Confluence OAuth token URL (optional)",
    )

    # ========================================================================
    # Document and Repository Configuration
    # ========================================================================

    document_paths: Optional[str] = Field(
        default=None,
        description="Comma-separated document paths (optional)",
    )

    code_repo_paths: Optional[str] = Field(
        default=None,
        description="Comma-separated code repository paths (optional)",
    )

    # ========================================================================
    # Embedding Model Configuration
    # ========================================================================

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for embeddings "
        "(default: sentence-transformers/all-MiniLM-L6-v2)",
    )

    # ========================================================================
    # Context and Conversation Settings
    # ========================================================================

    max_context_tokens: int = Field(
        default=2000,
        description="Maximum context tokens (default: 2000)",
    )

    max_conversation_history: int = Field(
        default=10,
        description="Maximum conversation history messages (default: 10)",
    )

    # ========================================================================
    # FastAPI Server Configuration
    # ========================================================================

    fastapi_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host (default: 0.0.0.0)",
    )

    fastapi_port: int = Field(
        default=8000,
        description="FastAPI server port (default: 8000)",
    )

    # ========================================================================
    # Field Validators
    # ========================================================================

    @field_validator("confluence_auth_type")
    @classmethod
    def validate_confluence_auth_type(cls, v: str) -> str:
        """Validate that confluence_auth_type is either 'api_token' or 'oauth'."""
        if v.lower() not in ["api_token", "oauth"]:
            raise ValueError("confluence_auth_type must be either 'api_token' or 'oauth'")
        return v.lower()

    @field_validator("use_embedded_weaviate", mode="before")
    @classmethod
    def validate_use_embedded_weaviate(cls, v) -> bool:
        """Convert string 'true'/'false' to boolean."""
        if isinstance(v, str):
            return v.lower() in ["true", "1", "yes"]
        return bool(v)

    @field_validator(
        "weaviate_port", "fastapi_port", "max_context_tokens", "max_conversation_history"
    )
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        """Validate that port and token numbers are positive integers."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("fastapi_host")
    @classmethod
    def validate_fastapi_host(cls, v: str) -> str:
        """Validate FastAPI host is a valid hostname or IP."""
        if not v:
            raise ValueError("fastapi_host cannot be empty")
        return v

    @field_validator("confluence_base_url")
    @classmethod
    def validate_confluence_url(cls, v: str) -> str:
        """Validate Confluence base URL is not empty."""
        if not v or not v.startswith("http"):
            raise ValueError("confluence_base_url must be a valid HTTP(S) URL")
        return v.rstrip("/")  # Remove trailing slash for consistency

    def __init__(self, **data):
        """Initialize settings and validate required fields.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        super().__init__(**data)

        # Conditional validation for Confluence auth when explicitly configured
        # If username is set, it means API token auth is intended
        if self.confluence_username and not self.confluence_api_token:
            raise ValueError("When confluence_username is set, confluence_api_token is required")

        # If API token is set, username should also be set
        if self.confluence_api_token and not self.confluence_username:
            raise ValueError("When confluence_api_token is set, confluence_username is required")

        # If OAuth client ID is set, other OAuth fields should be set
        if self.confluence_oauth_client_id:
            if not self.confluence_oauth_client_secret or not self.confluence_oauth_token_url:
                raise ValueError(
                    "When confluence_oauth_client_id is set, both "
                    "confluence_oauth_client_secret and confluence_oauth_token_url are required"
                )

    def get_document_paths_list(self) -> list[str]:
        """Parse comma-separated document paths into a list.

        Returns:
            List of document paths, or empty list if not configured.
        """
        if not self.document_paths:
            return []
        return [path.strip() for path in self.document_paths.split(",")]

    def get_code_repo_paths_list(self) -> list[str]:
        """Parse comma-separated code repo paths into a list.

        Returns:
            List of code repository paths, or empty list if not configured.
        """
        if not self.code_repo_paths:
            return []
        return [path.strip() for path in self.code_repo_paths.split(",")]
