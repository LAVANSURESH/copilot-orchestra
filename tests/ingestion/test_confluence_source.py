"""Tests for Confluence document source."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import base64
from src.ingestion.confluence_source import ConfluenceSource


class TestConfluenceSource:
    """Test Confluence document loading."""

    def test_confluence_api_token_auth(self):
        """Test API token auth header generation."""
        source = ConfluenceSource(
            base_url="https://example.atlassian.net",
            auth_type="api_token",
            space_keys=["TEST"],
            username="user@example.com",
            api_token="secret_token_12345"
        )
        
        headers = source._get_auth_headers()
        
        # Should have Authorization header
        assert "Authorization" in headers
        
        # Should be Basic auth with base64 encoded username:token
        auth_value = headers["Authorization"]
        assert auth_value.startswith("Basic ")
        
        # Decode and verify
        encoded = auth_value.replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode('utf-8')
        assert decoded == "user@example.com:secret_token_12345"

    def test_confluence_oauth_auth(self):
        """Test OAuth auth setup."""
        # Mock the OAuth token request
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"access_token": "oauth_token_xyz"}
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            source = ConfluenceSource(
                base_url="https://example.atlassian.net",
                auth_type="oauth",
                space_keys=["TEST"],
                client_id="client123",
                client_secret="secret123"
            )
            
            headers = source._get_auth_headers()
            
            # Should have Authorization header with Bearer token
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Bearer ")

    def test_fetch_pages_with_pagination(self):
        """Test pagination handling with mocked responses."""
        with patch('requests.get') as mock_get:
            # Mock first page response (25 pages, has more)
            page1_response = Mock()
            page1_response.json.return_value = {
                "results": [
                    {
                        "id": str(i),
                        "title": f"Page {i}",
                        "body": {"storage": {"value": f"<p>Content {i}</p>"}},
                    }
                    for i in range(25)
                ],
                "start": 0,
                "limit": 25,
                "size": 25,
                "isLastPage": False,
            }
            page1_response.status_code = 200
            
            # Mock second page response (5 pages, last page)
            page2_response = Mock()
            page2_response.json.return_value = {
                "results": [
                    {
                        "id": str(i),
                        "title": f"Page {i}",
                        "body": {"storage": {"value": f"<p>Content {i}</p>"}},
                    }
                    for i in range(25, 30)
                ],
                "start": 25,
                "limit": 25,
                "size": 5,
                "isLastPage": True,
            }
            page2_response.status_code = 200
            
            mock_get.side_effect = [page1_response, page2_response]
            
            source = ConfluenceSource(
                base_url="https://example.atlassian.net",
                auth_type="api_token",
                space_keys=["TEST"],
                username="user@example.com",
                api_token="token"
            )
            
            documents = source.load()
            
            # Should have all 30 documents
            assert len(documents) == 30
            assert all(doc.source_type == "confluence" for doc in documents)

    def test_parse_confluence_page(self):
        """Test page parsing and HTML stripping."""
        source = ConfluenceSource(
            base_url="https://example.atlassian.net",
            auth_type="api_token",
            space_keys=["TEST"],
            username="user@example.com",
            api_token="token"
        )
        
        page = {
            "id": "12345",
            "title": "Test Page",
            "body": {
                "storage": {
                    "value": "<p>This is a <strong>test</strong> page.</p><p>With HTML tags.</p>"
                }
            },
            "links": {
                "webui": "/wiki/spaces/TEST/pages/12345/Test+Page"
            }
        }
        
        doc = source._parse_confluence_page(page)
        
        # Should strip HTML and extract text
        assert "This is a test page." in doc.content or "test" in doc.content.lower()
        assert "<p>" not in doc.content
        assert "<strong>" not in doc.content
        assert doc.title == "Test Page"
        assert doc.source_type == "confluence"
        assert doc.metadata["page_id"] == "12345"

    def test_fetch_space_pages(self):
        """Test fetching pages from a specific space."""
        with patch('requests.get') as mock_get:
            # Mock space pages response
            response = Mock()
            response.json.return_value = {
                "results": [
                    {
                        "id": "1",
                        "title": "Page 1",
                        "body": {"storage": {"value": "<p>Content 1</p>"}},
                        "links": {"webui": "/wiki/spaces/TEST/pages/1"}
                    },
                    {
                        "id": "2",
                        "title": "Page 2",
                        "body": {"storage": {"value": "<p>Content 2</p>"}},
                        "links": {"webui": "/wiki/spaces/TEST/pages/2"}
                    }
                ],
                "start": 0,
                "limit": 25,
                "size": 2,
                "isLastPage": True,
            }
            response.status_code = 200
            mock_get.return_value = response
            
            source = ConfluenceSource(
                base_url="https://example.atlassian.net",
                auth_type="api_token",
                space_keys=["TEST"],
                username="user@example.com",
                api_token="token"
            )
            
            pages = source._fetch_space_pages("TEST")
            
            # Should return list of page dicts
            assert len(pages) == 2
            assert all("id" in page for page in pages)
            assert all("title" in page for page in pages)

    def test_multiple_spaces(self):
        """Test loading from multiple Confluence spaces."""
        with patch('requests.get') as mock_get:
            # Create mock responses for each space
            def mock_get_response(url, *args, **kwargs):
                response = Mock()
                if "TEST1" in url:
                    response.json.return_value = {
                        "results": [
                            {
                                "id": "1",
                                "title": "Page 1",
                                "body": {"storage": {"value": "<p>Content 1</p>"}},
                                "links": {"webui": "/wiki/spaces/TEST1/pages/1"}
                            }
                        ],
                        "isLastPage": True,
                    }
                else:  # TEST2
                    response.json.return_value = {
                        "results": [
                            {
                                "id": "2",
                                "title": "Page 2",
                                "body": {"storage": {"value": "<p>Content 2</p>"}},
                                "links": {"webui": "/wiki/spaces/TEST2/pages/2"}
                            }
                        ],
                        "isLastPage": True,
                    }
                response.status_code = 200
                return response
            
            mock_get.side_effect = mock_get_response
            
            source = ConfluenceSource(
                base_url="https://example.atlassian.net",
                auth_type="api_token",
                space_keys=["TEST1", "TEST2"],
                username="user@example.com",
                api_token="token"
            )
            
            documents = source.load()
            
            # Should have documents from both spaces
            assert len(documents) == 2
