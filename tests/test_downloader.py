"""Tests for the document download service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.downloader import (
    _ALLOWED_EXTENSIONS,
    BinaryContentError,
    DownloadTooLargeError,
    UnsafeRedirectError,
    UnsupportedContentTypeError,
    UnsupportedExtensionError,
    _is_allowed_content_type,
    _looks_like_text,
    _ssrf_safe_redirect_handler,
    download_document,
)


@pytest.fixture(autouse=True)
def _mock_validate_url():
    """Prevent real DNS resolution in every download test."""
    with patch("app.services.downloader.validate_url"):
        yield


class TestDownloadDocument:
    """Tests for ``download_document``."""

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_successful_download(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Downloads text content and returns decoded string."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-length": "11",
            "content-type": "text/plain; charset=utf-8",
        }
        mock_response.charset_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [b"hello world"]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = download_document("https://example.com/doc.txt")

        assert result == "hello world"

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_oversized_content_length(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects early when Content-Length exceeds limit."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 100

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/plain",
            "content-length": "5000",
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = []
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            DownloadTooLargeError,
            match=r"Content-Length",
        ):
            download_document("https://example.com/big.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_oversized_streaming_body(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects when streamed bytes exceed limit."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 10

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/plain",
        }  # no content-length
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [
            b"A" * 20,
        ]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            DownloadTooLargeError,
            match=r"exceeded",
        ):
            download_document("https://example.com/big.txt")


class TestSsrfSafeRedirectHandler:
    """Tests for the ``_ssrf_safe_redirect_handler`` hook."""

    def test_allows_safe_redirect(self):
        """Redirect to a public URL passes without error."""
        request = MagicMock(spec=httpx.Request)
        response = MagicMock(spec=httpx.Response)
        next_req = MagicMock(spec=httpx.Request)
        next_req.url = httpx.URL(
            "https://cdn.example.com/doc.txt",
        )
        response.next_request = next_req

        with patch(
            "app.services.downloader.validate_url",
            return_value="ok",
        ):
            # Should not raise
            _ssrf_safe_redirect_handler(request, response)

    def test_blocks_redirect_to_private_ip(self):
        """Redirect to a private IP raises UnsafeRedirectError."""
        request = MagicMock(spec=httpx.Request)
        response = MagicMock(spec=httpx.Response)
        next_req = MagicMock(spec=httpx.Request)
        next_req.url = httpx.URL("http://169.254.169.254/meta")
        response.next_request = next_req

        with (
            patch(
                "app.services.downloader.validate_url",
                side_effect=ValueError("private IP"),
            ),
            pytest.raises(
                UnsafeRedirectError,
                match="blocked by SSRF",
            ),
        ):
            _ssrf_safe_redirect_handler(request, response)

    def test_no_redirect_is_noop(self):
        """Non-redirect response (next_request is None) passes."""
        request = MagicMock(spec=httpx.Request)
        response = MagicMock(spec=httpx.Response)
        response.next_request = None

        # Should not raise
        _ssrf_safe_redirect_handler(request, response)


class TestContentTypeValidation:
    """Tests for Content-Type validation in ``download_document``."""

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_pdf_content_type(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects response with application/pdf Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "application/pdf",
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            UnsupportedContentTypeError,
            match=r"Unsupported Content-Type",
        ):
            download_document("https://example.com/doc.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_image_content_type(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects response with image/* Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "image/png",
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            UnsupportedContentTypeError,
            match=r"Unsupported Content-Type",
        ):
            download_document("https://example.com/photo.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_accepts_text_plain(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Accepts text/plain Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/plain; charset=utf-8",
            "content-length": "5",
        }
        mock_response.charset_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [b"hello"]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = download_document("https://example.com/doc.txt")
        assert result == "hello"

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_accepts_text_markdown(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Accepts text/markdown Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/markdown",
            "content-length": "7",
        }
        mock_response.charset_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [b"# Hello"]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = download_document("https://example.com/doc.md")
        assert result == "# Hello"

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_missing_content_type(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects response with no Content-Type header."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {}  # no content-type
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            UnsupportedContentTypeError,
            match=r"Unsupported Content-Type",
        ):
            download_document("https://example.com/doc.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_octet_stream(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects application/octet-stream Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "application/octet-stream",
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            UnsupportedContentTypeError,
            match=r"Unsupported Content-Type",
        ):
            download_document("https://example.com/doc.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_application_json(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Rejects application/json Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "application/json",
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            UnsupportedContentTypeError,
            match=r"Unsupported Content-Type",
        ):
            download_document("https://example.com/data.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_accepts_text_x_markdown(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """Accepts text/x-markdown Content-Type."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/x-markdown",
            "content-length": "7",
        }
        mock_response.charset_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [b"# Hello"]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = download_document("https://example.com/doc.md")
        assert result == "# Hello"


class TestIsAllowedContentType:
    """Tests for the ``_is_allowed_content_type`` helper."""

    def test_accepts_text_plain(self):
        """text/plain is allowed."""
        assert _is_allowed_content_type("text/plain") is True

    def test_accepts_text_plain_with_charset(self):
        """text/plain with charset parameter is allowed."""
        assert (
            _is_allowed_content_type(
                "text/plain; charset=utf-8",
            )
            is True
        )

    def test_accepts_text_markdown(self):
        """text/markdown is allowed."""
        assert _is_allowed_content_type("text/markdown") is True

    def test_accepts_application_markdown(self):
        """application/markdown is allowed."""
        assert _is_allowed_content_type("application/markdown") is True

    def test_rejects_application_pdf(self):
        """application/pdf is not allowed."""
        assert _is_allowed_content_type("application/pdf") is False

    def test_rejects_octet_stream(self):
        """application/octet-stream is not allowed."""
        assert (
            _is_allowed_content_type(
                "application/octet-stream",
            )
            is False
        )

    def test_rejects_none(self):
        """None Content-Type is not allowed."""
        assert _is_allowed_content_type(None) is False

    def test_rejects_empty_string(self):
        """Empty string Content-Type is not allowed."""
        assert _is_allowed_content_type("") is False

    def test_rejects_application_json(self):
        """application/json is not allowed."""
        assert _is_allowed_content_type("application/json") is False


class TestLooksLikeText:
    """Tests for the ``_looks_like_text`` byte-sniff helper."""

    def test_plain_ascii(self):
        """Normal ASCII text passes."""
        assert _looks_like_text(b"Hello, world!") is True

    def test_utf8_text(self):
        """UTF-8 text passes."""
        assert _looks_like_text("Héllo wörld".encode()) is True

    def test_rejects_pdf_signature(self):
        """PDF magic bytes are rejected."""
        assert _looks_like_text(b"%PDF-1.7 ...") is False

    def test_rejects_zip_signature(self):
        """ZIP/DOCX magic bytes are rejected."""
        assert _looks_like_text(b"PK\x03\x04 ...") is False

    def test_rejects_png_signature(self):
        """PNG magic bytes are rejected."""
        assert _looks_like_text(b"\x89PNG\r\n\x1a\n") is False

    def test_rejects_jpeg_signature(self):
        """JPEG magic bytes are rejected."""
        assert _looks_like_text(b"\xff\xd8\xff\xe0") is False

    def test_rejects_null_bytes(self):
        """Data containing null bytes is rejected."""
        assert _looks_like_text(b"hello\x00world") is False

    def test_rejects_gif_signature(self):
        """GIF magic bytes are rejected."""
        assert _looks_like_text(b"GIF89a\x01\x00") is False

    def test_rejects_gzip_signature(self):
        """GZIP magic bytes are rejected."""
        assert _looks_like_text(b"\x1f\x8b\x08\x00") is False

    def test_empty_bytes(self):
        """Empty data passes (no binary indicators)."""
        assert _looks_like_text(b"") is True

    def test_markdown_content(self):
        """Markdown content passes."""
        md = b"# Title\n\nSome **bold** text\n"
        assert _looks_like_text(md) is True


class TestByteSniffIntegration:
    """Integration tests: byte-sniff catches lying Content-Type."""

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_pdf_disguised_as_text(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """PDF body with text/plain Content-Type is rejected."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/plain",
            "content-length": "100",
        }
        mock_response.charset_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [
            b"%PDF-1.7 fake pdf content here",
        ]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            BinaryContentError,
            match=r"binary",
        ):
            download_document("https://example.com/doc.txt")

    @patch("app.services.downloader.get_settings")
    @patch("app.services.downloader.httpx.Client")
    def test_rejects_zip_disguised_as_text(
        self,
        mock_client_cls,
        mock_gs,
    ):
        """ZIP body with text/plain Content-Type is rejected."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "text/plain",
            "content-length": "50",
        }
        mock_response.charset_encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = [
            b"PK\x03\x04 fake zip",
        ]
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = lambda s: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(
            BinaryContentError,
            match=r"binary",
        ):
            download_document("https://example.com/doc.txt")


class TestExtensionGuardrail:
    """Tests for the URL extension guardrail."""

    def test_allows_txt_extension(self):
        """.txt is in the allowed set."""
        assert ".txt" in _ALLOWED_EXTENSIONS

    def test_allows_md_extension(self):
        """.md is in the allowed set."""
        assert ".md" in _ALLOWED_EXTENSIONS

    def test_allows_markdown_extension(self):
        """.markdown is in the allowed set."""
        assert ".markdown" in _ALLOWED_EXTENSIONS

    def test_rejects_no_extension(self):
        """URLs without an extension are rejected."""
        assert "" not in _ALLOWED_EXTENSIONS

    @patch("app.services.downloader.get_settings")
    def test_rejects_extensionless_url(self, mock_gs):
        """URL with no file extension is rejected early."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        with pytest.raises(
            UnsupportedExtensionError,
            match=r"<none>",
        ):
            download_document("https://example.com/document")

    @patch("app.services.downloader.get_settings")
    def test_rejects_pdf_extension(self, mock_gs):
        """URL with .pdf extension is rejected early."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        with pytest.raises(
            UnsupportedExtensionError,
            match=r"\.pdf",
        ):
            download_document("https://example.com/doc.pdf")

    @patch("app.services.downloader.get_settings")
    def test_rejects_docx_extension(self, mock_gs):
        """URL with .docx extension is rejected early."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        with pytest.raises(
            UnsupportedExtensionError,
            match=r"\.docx",
        ):
            download_document("https://example.com/doc.docx")

    @patch("app.services.downloader.get_settings")
    def test_rejects_json_extension(self, mock_gs):
        """URL with .json extension is rejected early."""
        mock_gs.return_value.DOC_DOWNLOAD_TIMEOUT = 30
        mock_gs.return_value.DOC_DOWNLOAD_MAX_BYTES = 1_000_000

        with pytest.raises(
            UnsupportedExtensionError,
            match=r"\.json",
        ):
            download_document("https://example.com/data.json")


class TestValidateUrlDefenceInDepth:
    """Tests for the defence-in-depth URL re-validation."""

    def test_rejects_private_ip_url(self):
        """URL pointing to a private IP is rejected."""
        with (
            patch(
                "app.services.downloader.validate_url",
                side_effect=ValueError("private IP"),
            ),
            pytest.raises(
                ValueError,
                match="private IP",
            ),
        ):
            download_document("http://169.254.169.254/meta")
