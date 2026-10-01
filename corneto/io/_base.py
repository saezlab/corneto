"""Private stdlib-only adapters shared by CORNETO file readers."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import math
import numbers
import os
import re
import tempfile
import urllib.parse
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Generator, TextIO

_COMPRESSION = {"auto", None, "gzip", "bz2", "xz"}
_MAGIC = ((b"\x1f\x8b", "gzip"), (b"BZh", "bz2"), (b"\xfd7zXZ\x00", "xz"))
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_COPY_CHUNK_SIZE = 1024 * 1024
_SPOOL_MAX_SIZE = 8 * 1024 * 1024


def _validate_options(compression: str | None, timeout: float) -> None:
    if compression is not None and (not isinstance(compression, str) or compression not in _COMPRESSION):
        raise ValueError("compression must be 'auto', None, 'gzip', 'bz2', or 'xz'")
    if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real):
        raise ValueError("timeout must be a finite positive number")
    try:
        valid_timeout = math.isfinite(float(timeout)) and timeout > 0
    except (OverflowError, TypeError, ValueError):
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("timeout must be a finite positive number")


def _validate_http_url(url: str) -> urllib.parse.SplitResult:
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("HTTP(S) URLs cannot contain control characters")
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; only HTTP and HTTPS are supported")
    if not parsed.netloc or not hostname:
        raise ValueError(f"Invalid HTTP(S) URL with no hostname: {url!r}")
    if any(char.isspace() for char in hostname):
        raise ValueError(f"Invalid hostname in URL: {url!r}")
    try:
        hostname.encode("idna")
    except UnicodeError as exc:
        raise ValueError(f"Invalid hostname in URL: {url!r}") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"Invalid port in URL: {url!r}")
    return parsed


def _source_kind(source: object) -> str:
    if isinstance(source, os.PathLike):
        return "path"
    if hasattr(source, "read"):
        return "stream"
    if not isinstance(source, str):
        raise TypeError("source must be a path, HTTP(S) URL, or readable stream")
    if _WINDOWS_DRIVE.match(source):
        return "path"
    try:
        parsed = urllib.parse.urlsplit(source)
    except ValueError as exc:
        raise ValueError(f"Invalid source URL: {source!r}") from exc
    if parsed.scheme:
        _validate_http_url(source)
        return "url"
    return "path"


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep urllib redirects within the supported HTTP(S) schemes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(request: urllib.request.Request, timeout: float):
    """Open a request with the standard urllib handlers and safe redirects."""
    return urllib.request.build_opener(_RedirectHandler()).open(request, timeout=timeout)


class _CountingResponse:
    """Expose a response stream and report a short body once EOF is observed."""

    def __init__(self, response, content_length: int | None):
        self._response = response
        self._content_length = content_length
        self._count = 0
        self._checked_eof = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._response.read(size)
        if data:
            self._count += len(data)
        elif size != 0 and not self._checked_eof:
            self._checked_eof = True
            if self._content_length is not None and self._count < self._content_length:
                raise EOFError(
                    "HTTP response ended before Content-Length bytes were received "
                    f"({self._count} of {self._content_length})"
                )
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        return size


class _DelegatingRaw(io.RawIOBase):
    """Raw adapter that never closes its wrapped stream."""

    def __init__(self, source, prefix: bytes = b"", *, disable_seeking: bool = False):
        super().__init__()
        self._source = source
        self._prefix = memoryview(prefix)
        self._prefix_offset = 0
        self._disable_seeking = disable_seeking

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        if self._disable_seeking:
            return False
        try:
            return bool(self._source.seekable())
        except (AttributeError, OSError):
            return False

    def tell(self) -> int:
        if self._prefix_offset < len(self._prefix):
            raise OSError("cannot tell while buffered prefix bytes remain")
        return self._source.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if self._disable_seeking:
            raise io.UnsupportedOperation("stream is not seekable")
        if self._prefix_offset < len(self._prefix):
            raise OSError("cannot seek while buffered prefix bytes remain")
        return self._source.seek(offset, whence)

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        prefix_remaining = len(self._prefix) - self._prefix_offset
        if prefix_remaining:
            size = min(len(view), prefix_remaining)
            view[:size] = self._prefix[self._prefix_offset : self._prefix_offset + size]
            self._prefix_offset += size
            if size == len(view):
                return size
            tail = self._source.read(len(view) - size)
            if isinstance(tail, str):
                raise TypeError("binary loaders require a binary readable stream")
            if tail:
                view[size : size + len(tail)] = tail
            return size + len(tail)
        data = self._source.read(len(view))
        if isinstance(data, str):
            raise TypeError("binary loaders require a binary readable stream")
        size = len(data)
        view[:size] = data
        return size


def _as_buffered(source, *, prefix: bytes = b"", disable_seeking: bool = False) -> io.BufferedReader:
    return io.BufferedReader(_DelegatingRaw(source, prefix=prefix, disable_seeking=disable_seeking))


def _detect_compression(prefix: bytes) -> str | None:
    return next((name for magic, name in _MAGIC if prefix.startswith(magic)), None)


def _decoder(stream, compression: str | None, stack: ExitStack):
    if compression == "gzip":
        decoded = gzip.GzipFile(fileobj=stream, mode="rb")
    elif compression == "bz2":
        decoded = bz2.BZ2File(stream, mode="rb")
    elif compression == "xz":
        decoded = lzma.LZMAFile(stream, mode="rb")
    else:
        return stream
    stack.callback(decoded.close)
    return decoded


def _is_seekable(stream) -> bool:
    try:
        return bool(stream.seekable())
    except (AttributeError, OSError):
        return False


def _content_length(response) -> int | None:
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    if value is None:
        getter = getattr(response, "getheader", None)
        value = getter("Content-Length") if getter is not None else None
    if value is None:
        return None
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid HTTP Content-Length header: {value!r}") from exc
    if length < 0:
        raise ValueError(f"Invalid HTTP Content-Length header: {value!r}")
    return length


@contextmanager
def _open_binary(
    source,
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
    seekable: bool = False,
) -> Generator[BinaryIO, None, None]:
    """Open a path, HTTP(S) URL, or borrowed binary stream for reading.

    Compression is detected by magic bytes in auto mode. Borrowed streams are
    read from their current position and remain open after this context exits.
    """
    _validate_options(compression, timeout)
    if not isinstance(seekable, bool):
        raise TypeError("seekable must be a boolean")
    kind = _source_kind(source)
    with ExitStack() as stack:
        if kind == "path":
            raw = stack.enter_context(Path(source).open("rb"))
            original_seekable = _is_seekable(raw)
        elif kind == "url":
            request = urllib.request.Request(
                source,
                headers={"User-Agent": "corneto", "Accept-Encoding": "identity"},
            )
            response = stack.enter_context(_open_url(request, float(timeout)))
            transport_encoding = (
                getattr(response, "headers", {}).get("Content-Encoding", "identity") or "identity"
            ).lower()
            if transport_encoding not in {"", "identity", "gzip"}:
                raise ValueError(f"Unsupported HTTP Content-Encoding: {transport_encoding!r}")
            raw = _CountingResponse(response, _content_length(response))
            original_seekable = False
            if transport_encoding == "gzip":
                transport = gzip.GzipFile(fileobj=raw, mode="rb")
                stack.callback(transport.close)
                raw = transport
        else:
            raw = source
            if not callable(getattr(raw, "read", None)):
                raise TypeError("source must be a path, HTTP(S) URL, or readable stream")
            if isinstance(raw, io.TextIOBase):
                raise TypeError("binary loaders require a binary readable stream")
            try:
                sample_type = raw.read(0)
            except (AttributeError, OSError) as exc:
                raise TypeError("source must be a readable binary stream") from exc
            if isinstance(sample_type, str):
                raise TypeError("binary loaders require a binary readable stream")
            if not isinstance(sample_type, bytes):
                raise TypeError("binary loaders require a stream whose read() returns bytes")
            original_seekable = _is_seekable(raw)

        start = None
        if original_seekable:
            try:
                start = raw.tell()
            except (AttributeError, OSError):
                original_seekable = False
        prefix_parts = []
        prefix_size = 0
        while prefix_size < 6:
            chunk = raw.read(6 - prefix_size)
            if isinstance(chunk, str):
                raise TypeError("binary loaders require a binary readable stream")
            if not isinstance(chunk, bytes):
                raise TypeError("binary loaders require a stream whose read() returns bytes")
            if not chunk:
                break
            prefix_parts.append(chunk)
            prefix_size += len(chunk)
        prefix = b"".join(prefix_parts)

        if compression == "auto":
            selected_compression = _detect_compression(prefix)
        else:
            selected_compression = compression

        reuse_source = False
        if seekable and original_seekable and start is not None:
            # gzip's random-access reader rewinds the underlying stream to
            # absolute offset zero. Keep embedded compressed streams
            # forward-only and spool their decoded contents below.
            if selected_compression is None or start == 0:
                try:
                    raw.seek(start)
                    reuse_source = True
                except (AttributeError, OSError):
                    reuse_source = False

        if reuse_source:
            if selected_compression is None:
                decoded = raw
            else:
                buffered = stack.enter_context(_as_buffered(raw))
                decoded = _decoder(buffered, selected_compression, stack)
        else:
            buffered = stack.enter_context(
                _as_buffered(raw, prefix=prefix, disable_seeking=selected_compression is not None)
            )
            decoded = _decoder(buffered, selected_compression, stack)

        if seekable and (not reuse_source or not _is_seekable(decoded)):
            spool = stack.enter_context(tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_SIZE, mode="w+b"))
            while True:
                chunk = decoded.read(_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                spool.write(chunk)
            spool.seek(0)
            decoded = spool

        yield decoded


@contextmanager
def _open_text(
    source,
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Generator[TextIO, None, None]:
    """Open a text or binary source as text using the shared file contract."""
    _validate_options(compression, timeout)
    if _source_kind(source) == "stream":
        try:
            sample = source.read(0)
        except (AttributeError, OSError) as exc:
            raise TypeError("source must be a readable stream") from exc
        if isinstance(sample, str):
            if compression not in {"auto", None}:
                raise ValueError("compression cannot be applied to a borrowed text stream")
            yield source
            return
        if not isinstance(sample, bytes):
            raise TypeError("text loaders require a text or binary readable stream")

    with _open_binary(source, compression=compression, timeout=timeout) as binary:
        with io.TextIOWrapper(binary, encoding=encoding, newline=newline) as text:
            yield text


@contextmanager
def _as_local_path(
    source,
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
    suffix: str = "",
) -> Generator[Path, None, None]:
    """Materialize a decoded source in a temporary local file for path-only APIs.

    ``suffix`` names the materialized path for downstream readers; it does not
    convert the source format.
    """
    _validate_options(compression, timeout)
    with tempfile.TemporaryDirectory(prefix="corneto-io-") as directory:
        path = Path(directory) / f"source{suffix}"
        with _open_binary(source, compression=compression, timeout=timeout) as binary:
            with path.open("wb") as output:
                while True:
                    chunk = binary.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
        yield path
