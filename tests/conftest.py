from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))


@pytest.fixture(scope="session")
def reference_pdf(tmp_path_factory):
    import pymupdf

    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.insert_text((40, 50), "Page 1: alpha beta", fontsize=13, fontname="helv")
    page.insert_text((100, 120), "Rotated text", fontsize=11, fontname="cour", rotate=90)
    page.insert_textbox(
        pymupdf.Rect(40, 150, 350, 260),
        "one line\ntwo lines\nthree lines",
        fontsize=12,
        fontname="times-roman",
    )
    page = document.new_page(width=500, height=240)
    page.insert_text((30, 45), "Page 2: punctuation.pdf and a-b", fontsize=12)
    page.insert_text((30, 85), "Second line 123", fontsize=16, fontname="cour")
    data = document.tobytes(deflate=True, garbage=4)
    path = tmp_path_factory.mktemp("pdfs") / "reference.pdf"
    path.write_bytes(data)
    return data, path


@pytest.fixture(scope="session")
def unicode_pdf():
    import pymupdf

    if not shutil.which("fc-match"):
        pytest.skip("fontconfig not installed")
    match = subprocess.run(
        ["fc-match", "-f", "%{file}", "DejaVu Sans"],
        capture_output=True,
        text=True,
        check=False,
    )
    font = match.stdout.strip()
    if match.returncode or not font or not os.path.exists(font):
        pytest.skip("DejaVu Sans not installed")
    document = pymupdf.open()
    page = document.new_page(width=300, height=120)
    page.insert_font(fontname="dejavu", fontfile=font)
    page.insert_text((40, 50), "Café Ω Привет", fontname="dejavu", fontsize=12)
    return document.tobytes(deflate=True, garbage=4)


@pytest.fixture(scope="session")
def form_pdf():
    import pymupdf

    source = pymupdf.open()
    page = source.new_page(width=200, height=100)
    page.insert_text((20, 30), "Inside form")
    document = pymupdf.open()
    page = document.new_page(width=400, height=300)
    page.show_pdf_page(pymupdf.Rect(100, 100, 300, 200), source, 0)
    return document.tobytes(deflate=True, garbage=4)
