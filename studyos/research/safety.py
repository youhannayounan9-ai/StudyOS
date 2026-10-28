"""Network security, URL validation, and SSRF prevention for external research."""

from __future__ import annotations

import html
from html.parser import HTMLParser
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse
import httpcore
from httpcore._backends.sync import SyncBackend, SyncStream
import httpx

logger = logging.getLogger(__name__)

# Security boundaries
MAX_PAGE_BYTES = 512 * 1024  # 512 KB per external page
MAX_REDIRECTS = 3
DEFAULT_TIMEOUT_SECONDS = 10.0

BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "instance-data",
}

BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # Private RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),    # Private RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),   # Private RFC 1918
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local & cloud metadata (169.254.169.254)
    ipaddress.ip_network("100.64.0.0/10"),    # Carrier-grade NAT
    ipaddress.ip_network("fc00::/7"),         # IPv6 Unique Local
    ipaddress.ip_network("fe80::/10"),        # IPv6 Link-Local
    ipaddress.ip_network("::1/128"),          # IPv6 Loopback
]


class SSRFValidationError(ValueError):
    """Raised when a URL resolves to a forbidden private, loopback, or cloud-metadata destination."""
    pass


class SafeURLValidator:
    """Validates URLs and DNS resolution against SSRF and network leakage."""

    @classmethod
    def validate_url_scheme_and_host(cls, url: str) -> tuple[str, str, int | None]:
        """Validate URL scheme and extract host and port."""
        clean_url = (url or "").strip()
        if not clean_url:
            raise SSRFValidationError("URL cannot be empty.")

        parsed = urlparse(clean_url)
        if parsed.scheme not in ("http", "https"):
            raise SSRFValidationError(f"Forbidden URL scheme '{parsed.scheme}'. Only http and https are allowed.")

        host = parsed.hostname
        if not host:
            raise SSRFValidationError(f"Invalid URL: missing hostname in '{clean_url}'.")

        host_lower = host.lower()
        if host_lower in BLOCKED_HOSTNAMES:
            raise SSRFValidationError(f"Access to blocked host '{host}' is strictly forbidden.")

        return host_lower, parsed.scheme, parsed.port

    @classmethod
    def verify_destination_ip(cls, host: str, port: int | None = None) -> list[str]:
        """Resolve DNS and verify that no resolved IP addresses belong to private, loopback, or metadata ranges."""
        try:
            # Query DNS for IPv4 and IPv6 addresses
            addr_info = socket.getaddrinfo(host, port or 80, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise SSRFValidationError(f"DNS resolution failed for host '{host}': {e}") from e

        resolved_ips: list[str] = []
        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            resolved_ips.append(ip_str)
            try:
                ip_obj = ipaddress.ip_address(ip_str)
            except ValueError as e:
                raise SSRFValidationError(f"Invalid IP address format '{ip_str}': {e}") from e

            # Check loopback, link-local, private, multicast, reserved
            if ip_obj.is_loopback:
                raise SSRFValidationError(f"SSRF Alert: Host '{host}' resolves to loopback IP '{ip_str}'.")
            if ip_obj.is_link_local:
                raise SSRFValidationError(f"SSRF Alert: Host '{host}' resolves to link-local/cloud-metadata IP '{ip_str}'.")
            if ip_obj.is_private:
                raise SSRFValidationError(f"SSRF Alert: Host '{host}' resolves to private network IP '{ip_str}'.")
            if ip_obj.is_multicast or ip_obj.is_reserved or ip_obj.is_unspecified:
                raise SSRFValidationError(f"SSRF Alert: Host '{host}' resolves to special/reserved IP '{ip_str}'.")

            for blocked_net in BLOCKED_NETWORKS:
                if ip_obj in blocked_net:
                    raise SSRFValidationError(
                        f"SSRF Alert: Host '{host}' IP '{ip_str}' is inside blocked network '{blocked_net}'."
                    )

        if not resolved_ips:
            raise SSRFValidationError(f"No IP addresses resolved for host '{host}'.")

        return resolved_ips


class TextExtractor(HTMLParser):
    """Simple, secure HTML-to-text extractor without script or style execution."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore: bool = False
        self._ignore_tags = {"script", "style", "noscript", "svg", "header", "footer", "nav"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._ignore_tags:
            self._ignore = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignore_tags:
            self._ignore = False

    def handle_data(self, data: str) -> None:
        if not self._ignore:
            cleaned = data.strip()
            if cleaned:
                self._pieces.append(cleaned)

    def get_text(self) -> str:
        raw_text = " ".join(self._pieces)
        # Collapse excessive whitespace
        return re.sub(r"\s+", " ", raw_text).strip()


def extract_plain_text_from_html(html_content: str) -> str:
    """Parse HTML content securely into plain text."""
    parser = TextExtractor()
    parser.feed(html_content)
    return parser.get_text()


class SafeSyncBackend(SyncBackend):
    """Custom network backend that pins DNS resolution to validated safe IPs.

    Eliminates the TOCTOU DNS rebinding window by ensuring the TCP socket connects
    strictly to the verified IP address without a secondary unvalidated resolution.
    """

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.NetworkStream:
        safe_ips = SafeURLValidator.verify_destination_ip(host, port)
        pinned_ip = safe_ips[0]

        address = (pinned_ip, port)
        source_address = None if local_address is None else (local_address, 0)

        sock = socket.create_connection(
            address,
            timeout,
            source_address=source_address,
        )
        if socket_options:
            for option in socket_options:
                sock.setsockopt(*option)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return SyncStream(sock)


class SafeHTTPTransport(httpx.HTTPTransport):
    """HTTP transport configured with SafeSyncBackend for IP pinning."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ssl_context = self._pool._ssl_context
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_context,
            network_backend=SafeSyncBackend(),
            retries=self._pool._retries,
        )


class SafeExternalHttpClient:
    """HTTP client for external research that strictly enforces SSRF protection, redirect checking, and size bounds."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.transport = transport

    def fetch_url(self, url: str) -> tuple[str, str]:
        """Safely fetch an external URL. Returns (final_url, plain_text_content)."""
        current_url = url
        redirect_count = 0

        transport = self.transport if self.transport is not None else SafeHTTPTransport()
        with httpx.Client(
            transport=transport,
            follow_redirects=False,
            timeout=self.timeout,
            trust_env=False,  # Ignore system proxies
            headers={"User-Agent": "StudyOS-ResearchBot/1.0 (+https://github.com/studyos/studyos)"},
        ) as client:
            while True:
                # 1. Validate scheme and hostname
                host, scheme, port = SafeURLValidator.validate_url_scheme_and_host(current_url)

                # 2. Verify IP addresses via DNS
                SafeURLValidator.verify_destination_ip(host, port)

                # 3. Stream request with byte limit
                try:
                    response = client.get(current_url)
                except httpx.RequestError as exc:
                    raise IOError(f"Network error fetching external URL '{current_url}': {exc}") from exc

                # 4. Handle redirects
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_count += 1
                    if redirect_count > MAX_REDIRECTS:
                        raise SSRFValidationError(f"Too many redirects (exceeded limit of {MAX_REDIRECTS}).")
                    location = response.headers.get("Location")
                    if not location:
                        raise SSRFValidationError("Redirect response missing 'Location' header.")
                    # Resolve relative redirect
                    current_url = urljoin(current_url, location)
                    logger.debug(f"Following safe redirect {redirect_count} to '{current_url}'")
                    continue

                if response.status_code != 200:
                    raise IOError(f"External HTTP request returned status {response.status_code} for '{current_url}'.")

                # 5. Check content size bound
                raw_bytes = response.content
                if len(raw_bytes) > MAX_PAGE_BYTES:
                    # Truncate content safely
                    raw_bytes = raw_bytes[:MAX_PAGE_BYTES]

                # Decode text
                content_type = response.headers.get("Content-Type", "")
                encoding = response.encoding or "utf-8"
                try:
                    text_body = raw_bytes.decode(encoding, errors="replace")
                except Exception:
                    text_body = raw_bytes.decode("utf-8", errors="replace")

                # Parse plain text if HTML
                if "html" in content_type.lower() or "<html" in text_body[:500].lower():
                    plain_text = extract_plain_text_from_html(text_body)
                else:
                    plain_text = text_body.strip()

                return current_url, plain_text
