"""Tests for shared file, URL, stream, and compression adapters."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import threading
import urllib.error
import xml.etree.ElementTree as ET
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from corneto.graph import Graph
from corneto.io import _base, import_cobra_model, import_miom_model, load_graph_from_sif, read_sbml
from corneto.io._metabolism import _load_compressed_gem


class ForwardOnly:
    """Small-read binary source with no seek API."""

    def __init__(self, data: bytes, read_size: int = 2):
        self._stream = io.BytesIO(data)
        self._read_size = read_size
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            size = len(self._stream.getbuffer())
        return self._stream.read(min(size, self._read_size))

    def close(self):
        self.closed = True
        self._stream.close()


class Response(io.BytesIO):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None):
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def http_server():
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, headers, body = routes.get(self.path, (404, {}, b"missing"))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            if not any(name.lower() == "content-length" for name in headers):
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.routes = routes
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", routes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    ("codec", "compress"),
    [("gzip", gzip.compress), ("bz2", bz2.compress), ("xz", lzma.compress)],
)
def test_local_paths_auto_explicit_and_disabled_compression(tmp_path, codec, compress):
    payload = b"extensionless compressed bytes" * 100
    compressed_path = tmp_path / "source"
    compressed_path.write_bytes(compress(payload))

    with _base._open_binary(compressed_path) as stream:
        assert stream.read() == payload
    with _base._open_binary(compressed_path, compression=codec) as stream:
        assert stream.read() == payload
    with _base._open_binary(compressed_path, compression=None) as stream:
        assert stream.read() == compress(payload)


def test_stream_position_short_reads_and_borrowed_ownership():
    payload = b"forward-only gzip input"
    source = ForwardOnly(gzip.compress(payload))

    with _base._open_binary(source) as stream:
        assert stream.read() == payload

    assert not source.closed


@pytest.mark.parametrize("compress", [gzip.compress, bz2.compress, lzma.compress])
def test_seekable_compressed_stream_at_nonzero_position_is_spooled(compress):
    payload = b"archive body at an embedded offset"
    source = io.BytesIO(b"prefix" + compress(payload))
    source.seek(len(b"prefix"))

    with _base._open_binary(source, seekable=True) as decoded:
        assert decoded.seekable()
        assert decoded.read() == payload

    assert not source.closed


def test_seekable_nonseekable_stream_spools_and_rolls_over():
    payload = b"x" * (8 * 1024 * 1024 + 1)
    source = ForwardOnly(payload, read_size=64 * 1024)

    with _base._open_binary(source, seekable=True) as decoded:
        assert decoded.seekable()
        assert decoded.read() == payload
        assert decoded._rolled

    assert not source.closed


def test_text_adapters_accept_text_and_binary_streams_without_closing_them():
    text_source = io.StringIO("A\t1\tB\n")
    assert load_graph_from_sif(text_source).num_edges == 1
    assert not text_source.closed

    binary_source = io.BytesIO(b"A\t1\tB\n")
    assert load_graph_from_sif(binary_source).num_edges == 1
    assert not binary_source.closed

    with pytest.raises(ValueError, match="borrowed text stream"):
        with _base._open_text(text_source, compression="gzip"):
            pass
    assert not text_source.closed


def test_binary_adapter_rejects_text_stream_and_preserves_stream_on_errors():
    text_source = io.StringIO("not binary")
    with pytest.raises(TypeError, match="binary readable stream"):
        with _base._open_binary(text_source):
            pass
    assert not text_source.closed

    binary_source = io.BytesIO(b"invalid gzip")
    with pytest.raises((gzip.BadGzipFile, EOFError)):
        with _base._open_binary(binary_source, compression="gzip") as stream:
            stream.read()
    assert not binary_source.closed


def test_as_local_path_decodes_and_cleans_up_after_use():
    payload = b"path-only API payload"
    source = io.BytesIO(bz2.compress(payload))
    with _base._as_local_path(source, suffix=".xml") as path:
        assert path.suffix == ".xml"
        assert path.read_bytes() == payload
        saved_path = path
    assert not source.closed
    assert not saved_path.exists()


def test_as_local_path_cleanup_runs_when_consumer_raises():
    saved_path = None
    with pytest.raises(RuntimeError, match="consumer failure"):
        with _base._as_local_path(io.BytesIO(b"contents"), suffix=".xml") as path:
            saved_path = path
            assert path.read_bytes() == b"contents"
            raise RuntimeError("consumer failure")
    assert saved_path is not None
    assert not saved_path.exists()


def test_http_urls_magic_compression_query_redirect_and_request_headers(http_server):
    base_url, routes = http_server
    payload = b"remote content" * 20
    sif_payload = b"A\t1\tB\n"
    routes["/extensionless?version=1"] = (200, {}, gzip.compress(payload))
    routes["/redirect"] = (302, {"Location": "/extensionless?version=1"}, b"")
    routes["/sif?view=1"] = (200, {}, sif_payload)
    routes["/http-encoded-sif"] = (200, {"Content-Encoding": "gzip"}, gzip.compress(sif_payload))
    routes["/double-gzip-sif"] = (
        200,
        {"Content-Encoding": "gzip"},
        gzip.compress(gzip.compress(sif_payload)),
    )

    with _base._open_binary(f"{base_url}/extensionless?version=1") as stream:
        assert stream.read() == payload
    with _base._open_binary(f"{base_url}/redirect") as stream:
        assert stream.read() == payload
    assert load_graph_from_sif(f"{base_url}/sif?view=1").num_edges == 1
    assert Graph.from_sif(f"{base_url}/http-encoded-sif").num_edges == 1
    assert load_graph_from_sif(f"{base_url}/double-gzip-sif").num_edges == 1

    captured = {}

    def fake_open(request, timeout):
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return Response(payload, {"Content-Length": str(len(payload))})

    original = _base._open_url
    _base._open_url = fake_open
    try:
        with _base._open_binary("https://example.org/resource?format=raw", timeout=4.5) as stream:
            assert stream.read() == payload
    finally:
        _base._open_url = original

    assert captured["timeout"] == 4.5
    assert captured["headers"]["User-agent"] == "corneto"
    assert captured["headers"]["Accept-encoding"] == "identity"


def test_http_redirect_errors_encoding_and_truncated_content_length(http_server):
    base_url, routes = http_server
    routes["/bad-redirect"] = (302, {"Location": "ftp://example.org/file"}, b"")
    routes["/unsupported-encoding"] = (200, {"Content-Encoding": "br"}, b"data")
    routes["/truncated"] = (200, {"Content-Length": "9"}, b"short")

    with pytest.raises(ValueError, match="only HTTP and HTTPS"):
        with _base._open_binary(f"{base_url}/bad-redirect"):
            pass
    with pytest.raises(ValueError, match="Content-Encoding"):
        with _base._open_binary(f"{base_url}/unsupported-encoding"):
            pass
    with pytest.raises((EOFError, OSError)):
        with _base._open_binary(f"{base_url}/truncated") as stream:
            stream.read()
    with pytest.raises(urllib.error.HTTPError):
        with _base._open_binary(f"{base_url}/missing"):
            pass


def test_http_response_closes_when_sbml_parser_fails(monkeypatch):
    response = Response(b"<broken", {"Content-Length": "7"})
    monkeypatch.setattr(_base, "_open_url", lambda _request, _timeout: response)

    with pytest.raises(ET.ParseError):
        read_sbml("https://example.org/broken.xml")

    assert response.closed


def test_cobra_model_imports_zip_from_http_url(http_server):
    base_url, routes = http_server
    xml_path = Path(__file__).parent / "data" / "mitocore_v1.01.xml"
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("model.xml", xml_path.read_bytes())
    routes["/model.zip?revision=1"] = (200, {}, archive_buffer.getvalue())

    graph = import_cobra_model(f"{base_url}/model.zip?revision=1")

    assert graph.num_vertices == 441
    assert graph.num_edges == 555


@pytest.mark.parametrize(
    "source",
    [
        "ftp://example.org/file",
        "file:///tmp/model.xml",
        "http:///missing-host",
        "http://example.org:99999/file",
        "http://example.org/line\nbreak",
    ],
)
def test_invalid_source_urls_are_rejected(source):
    with pytest.raises(ValueError):
        with _base._open_binary(source):
            pass


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"compression": "zip"}, "compression"),
        ({"compression": []}, "compression"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": float("inf")}, "timeout"),
        ({"timeout": True}, "timeout"),
    ],
)
def test_invalid_source_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        with _base._open_binary(io.BytesIO(b"data"), **kwargs):
            pass


def test_sif_both_entrypoints_use_shared_stream_and_compression_support(tmp_path):
    compressed = tmp_path / "network"
    compressed.write_bytes(gzip.compress(b"A\t1\tB\nB\t-1\tC\n"))

    loaded = load_graph_from_sif(compressed)
    graph_method = Graph.from_sif(compressed)
    assert loaded.num_edges == graph_method.num_edges == 2

    with pytest.raises(ValueError, match="Invalid SIF line"):
        load_graph_from_sif(io.StringIO("A\t1\n"))


def test_extensionless_http_miom_archive_and_model_object(http_server):
    base_url, routes = http_server
    archive_path = Path(__file__).parents[1] / "methods" / "data" / "mitocore.miom"
    compressed_archive = archive_path.read_bytes()
    routes["/archive?revision=1"] = (200, {}, compressed_archive)

    local_arrays = _load_compressed_gem(io.BytesIO(compressed_archive))
    remote_arrays = _load_compressed_gem(f"{base_url}/archive?revision=1")
    assert [array.shape for array in remote_arrays] == [array.shape for array in local_arrays]
    assert local_arrays[0].shape == (441, 555)

    reactions = np.array(
        [("R1", 1.0, 10.0, "gene", "")],
        dtype=[("id", "U8"), ("lb", "f8"), ("ub", "f8"), ("gpr", "U8"), ("name", "U8")],
    )
    metabolites = np.array([("A",), ("B",)], dtype=[("id", "U8")])
    model = type("Model", (), {"S": np.array([[-1.0], [1.0]]), "R": reactions, "M": metabolites})()
    assert import_miom_model(model).num_edges == 1
