#!/usr/bin/env python3
"""Validate the SHA256 of the pinned wasm-graphviz CDN asset."""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

from corneto.contrib._util import DEFAULT_WASM_GRAPHVIZ_JS_URL


def main() -> int:
    checksum_path = pathlib.Path(__file__).with_name("wasm_graphviz_sha256.txt")
    expected = checksum_path.read_text(encoding="utf-8").strip().lower()
    if not expected:
        print(f"Expected checksum file is empty: {checksum_path}", file=sys.stderr)
        return 1

    with urllib.request.urlopen(DEFAULT_WASM_GRAPHVIZ_JS_URL, timeout=30) as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()

    if actual != expected:
        print("WASM Graphviz checksum mismatch.", file=sys.stderr)
        print(f"URL:      {DEFAULT_WASM_GRAPHVIZ_JS_URL}", file=sys.stderr)
        print(f"Expected: {expected}", file=sys.stderr)
        print(f"Actual:   {actual}", file=sys.stderr)
        return 1

    print("WASM Graphviz checksum OK.")
    print(f"URL:      {DEFAULT_WASM_GRAPHVIZ_JS_URL}")
    print(f"SHA256:   {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
