"""Confluence document source."""
import requests
import base64
from typing import List, Dict, Any
from hashlib import md5
from html.parser import HTMLParser
from src.ingestion.base import DocumentSource
from src.models.document import Document


class HTMLStripper(HTMLParser):
    """Simple HTML tag stripper."""

    def __init__(self):
        """Initialize the stripper."""
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        """Handle text data."""
        self.text.append(d)

    def get_data(self) -> str:
        """Get the stripped text."""
        return ''.join(self.text)


def strip_html(html_content: str) -> str:
    """Strip HTML tags from content.

    Args:
        html_content: HTML string

    Returns:
        Plain text without HTML tags
    """
    stripper = HTMLStripper()
    try:
        stripper.feed(html_content)
        return stripper.get_data()
    except Exception:
        # Fallback: return original if parsing fails
        return html_content


class ConfluenceSource(DocumentSource):
    """Load documents from Confluence."""

    def __init__(
        self,
        base_url: str,
        auth_type: str,
        space_keys: List[str],
        **auth_kwargs,
    ):
        """Initialize Confluence source.

        Args:
            base_url: Confluence base URL (e.g., https://example.atlassian.net)
            auth_type: Authentication type: "api_token" or "oauth"
            space_keys: List of Confluence space keys to load
            **auth_kwargs: Authentication credentials
                For api_token: username, api_token
                For oauth: client_id, client_secret, (optional) oauth_token
        """
        self.base_url = base_url
        self.auth_type = auth_type
        self.space_keys = space_keys
        self.auth_kwargs = auth_kwargs

        # Cache for OAuth token
        self._oauth_token = None
        if auth_type == "oauth" and "oauth_token" in auth_kwargs:
            self._oauth_token = auth_kwargs["oauth_token"]

    def load(self) -> List[Document]:
        """Load all pages from configured Confluence spaces.

        Returns:
            List of Document objects
        """
        documents = []

        for space_key in self.space_keys:
            pages = self._fetch_space_pages(space_key)
            for page in pages:
                doc = self._parse_confluence_page(page)
                documents.append(doc)

        return documents

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Confluence API.

        Returns:
            Dictionary with Authorization header
        """
        headers = {}

        if self.auth_type == "api_token":
            username = self.auth_kwargs.get("username")
            api_token = self.auth_kwargs.get("api_token")

            # Create Basic auth header
            credentials = f"{username}:{api_token}"
            encoded = base64.b64encode(credentials.encode()).decode('utf-8')
            headers["Authorization"] = f"Basic {encoded}"

        elif self.auth_type == "oauth":
            # If no cached token, fetch one
            if not self._oauth_token:
                self._oauth_token = self._fetch_oauth_token()

            headers["Authorization"] = f"Bearer {self._oauth_token}"

        return headers

    def _fetch_oauth_token(self) -> str:
        """Fetch OAuth token from Confluence.

        Returns:
            OAuth token string
        """
        client_id = self.auth_kwargs.get("client_id")
        client_secret = self.auth_kwargs.get("client_secret")

        # Construct OAuth token URL
        token_url = f"{self.base_url}/oauth/token"

        # Request token
        response = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }
        )

        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            raise Exception(f"Failed to fetch OAuth token: {response.status_code}")

    def _fetch_space_pages(self, space_key: str) -> List[Dict[str, Any]]:
        """Fetch all pages from a Confluence space.

        Args:
            space_key: Confluence space key

        Returns:
            List of page dictionaries
        """
        pages = []
        start = 0

        while True:
            # Construct URL for fetching pages
            url = (
                f"{self.base_url}/wiki/rest/api/content"
                f"?spaceKey={space_key}"
                f"&type=page"
                f"&expand=body.storage"
                f"&start={start}"
                f"&limit=25"
            )

            headers = self._get_auth_headers()

            # Fetch pages
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Failed to fetch Confluence pages: {response.status_code}")

            data = response.json()
            pages.extend(data.get("results", []))

            # Check if there are more pages
            if data.get("isLastPage", True):
                break

            start = data.get("start", 0) + data.get("limit", 25)

        return pages

    def _parse_confluence_page(self, page: Dict[str, Any]) -> Document:
        """Convert Confluence page to Document.

        Args:
            page: Confluence page dictionary

        Returns:
            Document object
        """
        page_id = page.get("id")
        title = page.get("title", "")

        # Extract HTML content and strip tags
        html_content = page.get("body", {}).get("storage", {}).get("value", "")
        content = strip_html(html_content)

        # Get source URL
        webui_link = page.get("links", {}).get("webui", "")
        if webui_link:
            source = f"{self.base_url}{webui_link}"
        else:
            source = f"{self.base_url}/page/{page_id}"

        # Create document ID
        doc_id = md5(f"{source}:{title}".encode()).hexdigest()

        metadata = {
            "page_id": page_id,
            "space_key": page.get("space", {}).get("key", ""),
        }

        return Document(
            id=doc_id,
            content=content,
            title=title,
            source=source,
            source_type="confluence",
            metadata=metadata,
        )

    def get_source_id(self) -> str:
        """Get unique identifier for this source.

        Returns:
            String identifier
        """
        spaces_str = "|".join(sorted(self.space_keys))
        combined = f"{self.base_url}:{spaces_str}"
        return md5(combined.encode()).hexdigest()

