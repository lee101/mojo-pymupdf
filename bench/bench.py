"""Benchmark mojo-pymupdf against upstream PyMuPDF on identical PDFs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_pymupdf as mojo_pdf  # noqa: E402
from mojo_pymupdf import _lib  # noqa: E402
import pymupdf as upstream  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def make_document():
    document = upstream.open()
    for page_number in range(60):
        page = document.new_page(width=600, height=800)
        for line in range(30):
            page.insert_text(
                (35, 35 + line * 24),
                f"Page {page_number + 1:02d} line {line + 1:02d}: Mojo parses PDF text quickly and correctly.",
                fontsize=10,
            )
    return document.tobytes(deflate=True, garbage=4)


def make_dense_page():
    document = upstream.open()
    page = document.new_page(width=600, height=40_100)
    for line in range(2_000):
        page.insert_text(
            (30, 30 + line * 20),
            f"record {line:04d} alpha beta gamma delta 12345",
            fontsize=9,
        )
    return document.tobytes(deflate=True, garbage=4)


def python_lex_count(data: bytes) -> int:
    white = b"\x00\t\n\f\r "
    delimiters = b"()<>[]{}/%"
    count = 0
    i = 0
    while i < len(data):
        if data[i] in white:
            i += 1
            continue
        if data[i] == 37:
            while i < len(data) and data[i] not in b"\r\n":
                i += 1
            continue
        count += 1
        if data[i] == 40:
            i += 1
            depth = 1
            escaped = False
            while i < len(data) and depth:
                c = data[i]
                i += 1
                if escaped:
                    escaped = False
                elif c == 92:
                    escaped = True
                elif c == 40:
                    depth += 1
                elif c == 41:
                    depth -= 1
        elif data[i] == 60 and i + 1 < len(data) and data[i + 1] != 60:
            i = data.find(b">", i + 1)
            i = len(data) if i < 0 else i + 1
        elif data[i] in b"[]":
            i += 1
        else:
            if data[i] == 47:
                i += 1
            while i < len(data) and data[i] not in white + delimiters:
                i += 1
            if i < len(data) and data[i] in b"<>":
                i += 2 if data[i : i + 2] in (b"<<", b">>") else 1
    return count


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def row(name, ours, reference, reference_name):
    ratio = reference / ours
    status = "faster" if ratio >= 1 else "slower"
    ratio_text = f"{ratio:.3f}" if ratio < 0.1 else f"{ratio:.2f}"
    return f"| {name} | {ours * 1e3:.2f} ms | {reference * 1e3:.2f} ms | {ratio_text}x {status} | {reference_name} |"


def main():
    library_data = make_document()
    dense_data = make_dense_page()
    upstream_document = upstream.open(stream=library_data, filetype="pdf")
    mojo_document = mojo_pdf.open(stream=library_data, filetype="pdf")
    upstream_dense = upstream.open(stream=dense_data, filetype="pdf")
    mojo_dense = mojo_pdf.open(stream=dense_data, filetype="pdf")

    assert "".join(page.get_text() for page in mojo_document) == "".join(
        page.get_text() for page in upstream_document
    )
    assert [word[4] for word in mojo_dense[0].get_text("words")] == [
        word[4] for word in upstream_dense[0].get_text("words")
    ]

    cases = []
    ours, _ = timeit(lambda: mojo_pdf.open(stream=library_data), repeat=3)
    reference, _ = timeit(lambda: upstream.open(stream=library_data, filetype="pdf"), repeat=3)
    cases.append(("open 60-page PDF", ours, reference, "PyMuPDF"))

    ours, _ = timeit(lambda: "".join(page.get_text() for page in mojo_document), repeat=3)
    reference, _ = timeit(lambda: "".join(page.get_text() for page in upstream_document), repeat=3)
    cases.append(("text, 60 pages / 1,800 lines", ours, reference, "PyMuPDF"))

    ours, _ = timeit(lambda: mojo_dense[0].get_text("words"), repeat=3)
    reference, _ = timeit(lambda: upstream_dense[0].get_text("words"), repeat=3)
    cases.append(("words, one page / 2,000 lines", ours, reference, "PyMuPDF"))

    ours, _ = timeit(lambda: mojo_dense[0].get_text("rawdict"), repeat=3)
    reference, _ = timeit(lambda: upstream_dense[0].get_text("rawdict"), repeat=3)
    cases.append(("rawdict, one page / 2,000 lines", ours, reference, "PyMuPDF"))

    content = b"q BT /F1 12 Tf 1 0 0 1 20 40 Tm [(Hello) -20 <4d6f6a6f>] TJ ET Q\n" * 40_000
    mojo_count = len(_lib.lex(content)[0])
    python_count = python_lex_count(content)
    assert mojo_count == python_count
    ours, _ = timeit(lambda: _lib.lex(content), repeat=5)
    reference, _ = timeit(lambda: python_lex_count(content), repeat=3)
    cases.append(("content lex, 2.9 MB", ours, reference, "pure Python"))

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}, Python {platform.python_version()}")
    print()
    print("| case | mojo-pymupdf | reference | ratio | reference implementation |")
    print("| --- | ---: | ---: | ---: | --- |")
    for case in cases:
        print(row(*case))


if __name__ == "__main__":
    main()
