from __future__ import annotations

import io
import json

import numpy as np
import pytest

import mojo_pymupdf as mupdf
from mojo_pymupdf import _lib

import pymupdf as upstream


def documents(data):
    return upstream.open(stream=data, filetype="pdf"), mupdf.open(stream=data, filetype="pdf")


def boxes(items):
    return np.asarray([item[:4] for item in items], dtype=float)


def test_build_exports_real_mojo_library():
    handle = _lib.lib()
    assert handle.mpdf_lex
    assert handle.mpdf_decode_string


def test_content_lexer_tokenizes_pdf_grammar():
    data = b"% comment\nBT /F1 12 Tf [(a\\n) -20 <4243>] TJ ET"
    kinds, offsets, lengths = _lib.lex(data)
    tokens = [data[int(a) : int(a + b)] for a, b in zip(offsets, lengths)]
    assert tokens == [b"BT", b"F1", b"12", b"Tf", b"[", b"a\\n", b"-20", b"4243", b"]", b"TJ", b"ET"]
    assert kinds.tolist() == [1, 2, 1, 1, 5, 3, 1, 4, 6, 1, 1]


@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        (b"a\\nb\\053", 3, b"a\nb+"),
        (b"a\\\r\nb", 3, b"ab"),
        (b"48656c6c6f2", 4, b"Hello "),
        (b"43 61 66 e9", 4, b"Caf\xe9"),
    ],
)
def test_mojo_string_decoder_matches_pdf_rules(raw, kind, expected):
    assert _lib.decode_string(raw, kind) == expected


def test_ffi_wrappers_reject_invalid_types_and_shapes():
    with pytest.raises(TypeError):
        _lib.lex(bytearray(b"BT"))
    with pytest.raises(ValueError):
        _lib.decode_string(b"abc", 99)
    with pytest.raises(TypeError):
        _lib.layout_glyphs(
            np.array([1 + 2j]), np.array([1.0]), (1, 0, 0, 1, 0, 0), 0, 1, 0, 10
        )
    with pytest.raises(TypeError):
        _lib.layout_glyphs(
            np.array([2**53 + 1]), np.array([1]), (1, 0, 0, 1, 0, 0), 0, 1, 0, 10
        )
    with pytest.raises(ValueError):
        _lib.layout_glyphs(
            np.array([1.0]), np.array([1.0]), (1, 0, 0), 0, 1, 0, 10
        )


def test_exported_ffi_rejects_null_and_negative_buffer_contracts():
    handle = _lib.lib()
    assert handle.mpdf_lex(0, 1, 0, 0, 0, 0) == -1
    assert handle.mpdf_decode_string(0, -1, 3, 0) == -1
    assert handle.mpdf_layout_glyphs(0, 0, 0, -1, *([0.0] * 10)) == -1


def test_pdf_value_tokenizer_makes_progress_on_stray_delimiters():
    from mojo_pymupdf._pdf import parse_value

    assert parse_value(b"}") == "}"


def expected_geometry(positions, advances, matrix, low, high, rise, page_height):
    a, b, c, d, e, f = matrix
    x0 = positions
    x1 = positions + advances
    xs = np.stack(
        (
            a * x0 + c * low + e,
            a * x1 + c * low + e,
            a * x0 + c * high + e,
            a * x1 + c * high + e,
        )
    )
    ys = page_height - np.stack(
        (
            b * x0 + d * low + f,
            b * x1 + d * low + f,
            b * x0 + d * high + f,
            b * x1 + d * high + f,
        )
    )
    return np.vstack(
        (
            a * positions + c * rise + e,
            page_height - (b * positions + d * rise + f),
            xs.min(axis=0),
            ys.min(axis=0),
            xs.max(axis=0),
            ys.max(axis=0),
        )
    )


@pytest.mark.parametrize(
    "count",
    [0, 11, _lib.LAYOUT_PARALLEL_THRESHOLD - 1, _lib.LAYOUT_PARALLEL_THRESHOLD + 7],
)
def test_simd_glyph_layout_tail_and_parallel_threshold(count):
    advances = np.linspace(0.25, 1.75, count, dtype=np.float64)
    positions = np.empty(count, dtype=np.float64)
    if count:
        positions[0] = 0.0
        np.cumsum(advances[:-1], out=positions[1:])
    matrix = (0.8, 0.6, -0.2, 0.9, 11.0, -4.0)
    actual = _lib.layout_glyphs(positions, advances, matrix, -3.0, 8.0, 1.5, 100.0)
    expected = expected_geometry(
        positions, advances, matrix, -3.0, 8.0, 1.5, 100.0
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_open_path_stream_and_file_object(reference_pdf):
    data, path = reference_pdf
    assert len(mupdf.open(path)) == 2
    assert len(mupdf.open(stream=data, filetype="pdf")) == 2
    assert len(mupdf.open(stream=io.BytesIO(data))) == 2


def test_document_page_count_iteration_and_negative_index(reference_pdf):
    data, _ = reference_pdf
    document = mupdf.open(stream=data)
    assert document.page_count == 2
    assert [page.number for page in document] == [0, 1]
    assert document[-1].number == 1
    with pytest.raises(ValueError):
        document.load_page(2)


def test_page_rect_matches_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    for left, right in zip(reference, ours):
        assert tuple(right.rect) == pytest.approx(tuple(left.rect))
        assert tuple(right.bound()) == pytest.approx(tuple(left.bound()))


def test_plain_text_matches_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    assert [page.get_text() for page in ours] == [page.get_text() for page in reference]


def test_document_get_page_text_matches_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    assert ours.get_page_text(1) == reference.get_page_text(1)


def test_words_text_and_indices_match_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    for left, right in zip(reference, ours):
        expected, actual = left.get_text("words"), right.get_text("words")
        assert [item[4:] for item in actual] == [item[4:] for item in expected]


def test_word_boxes_match_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    for left, right in zip(reference, ours):
        np.testing.assert_allclose(boxes(right.get_text("words")), boxes(left.get_text("words")), atol=5e-4)


def test_blocks_match_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    for left, right in zip(reference, ours):
        expected, actual = left.get_text("blocks"), right.get_text("blocks")
        assert [item[4:] for item in actual] == [item[4:] for item in expected]
        np.testing.assert_allclose(boxes(actual), boxes(expected), atol=5e-4)


def test_dict_structure_and_spans_match_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    expected = reference[0].get_text("dict")
    actual = ours[0].get_text("dict")
    assert actual["width"] == expected["width"]
    assert actual["height"] == expected["height"]
    assert len(actual["blocks"]) == len(expected["blocks"])
    expected_spans = [span for block in expected["blocks"] for line in block["lines"] for span in line["spans"]]
    actual_spans = [span for block in actual["blocks"] for line in block["lines"] for span in line["spans"]]
    assert [(s["text"], s["font"], s["flags"]) for s in actual_spans] == [
        (s["text"], s["font"], s["flags"]) for s in expected_spans
    ]
    np.testing.assert_allclose(
        [s["bbox"] for s in actual_spans], [s["bbox"] for s in expected_spans], atol=5e-4
    )


def test_rotated_line_direction_matches_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    expected = [line["dir"] for block in reference[0].get_text("dict")["blocks"] for line in block["lines"]]
    actual = [line["dir"] for block in ours[0].get_text("dict")["blocks"] for line in block["lines"]]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_rawdict_characters_and_boxes_match_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    expected = reference[0].get_text("rawdict")
    actual = ours[0].get_text("rawdict")
    e_chars = [char for block in expected["blocks"] for line in block["lines"] for span in line["spans"] for char in span["chars"]]
    a_chars = [char for block in actual["blocks"] for line in block["lines"] for span in line["spans"] for char in span["chars"]]
    assert [char["c"] for char in a_chars] == [char["c"] for char in e_chars]
    np.testing.assert_allclose([char["bbox"] for char in a_chars], [char["bbox"] for char in e_chars], atol=5e-4)
    np.testing.assert_allclose([char["origin"] for char in a_chars], [char["origin"] for char in e_chars], atol=5e-4)


@pytest.mark.parametrize("mode", ["json", "rawjson"])
def test_json_modes_are_semantically_the_dict_modes(reference_pdf, mode):
    document = mupdf.open(stream=reference_pdf[0])
    decoded = json.loads(document[0].get_text(mode))
    direct = document[0].get_text(mode.removesuffix("json") + "dict")
    assert decoded["width"] == direct["width"]
    assert len(decoded["blocks"]) == len(direct["blocks"])


def test_textpage_reuses_extraction(reference_pdf):
    document = mupdf.open(stream=reference_pdf[0])
    textpage = document[0].get_textpage()
    assert textpage is document[0].get_textpage()
    assert isinstance(textpage, mupdf.TextPage)
    assert textpage.extractText() == document[0].get_text()
    assert textpage.extractWORDS() == document[0].get_text("words")
    assert textpage.extractBLOCKS() == document[0].get_text("blocks")


def test_cached_parse_and_words_do_not_share_mutable_results(reference_pdf):
    first = mupdf.open(stream=reference_pdf[0])
    second = mupdf.open(stream=reference_pdf[0])
    assert first._pdf is second._pdf
    first.close()
    words = second[0].get_text("words")
    count = len(words)
    words.clear()
    assert len(second[0].get_text("words")) == count


def test_rawdict_results_do_not_share_mutable_containers(reference_pdf):
    page = mupdf.open(stream=reference_pdf[0])[0]
    first = page.get_text("rawdict")
    expected = page.get_text("rawdict")
    first["blocks"][0]["lines"][0]["spans"][0]["chars"][0]["c"] = "changed"
    first["blocks"].clear()
    assert page.get_text("rawdict") == expected


def test_clip_matches_upstream_for_whole_line(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    clip = (20, 25, 180, 60)
    assert ours[0].get_text(clip=clip) == reference[0].get_text(clip=clip)


def test_word_delimiters_match_upstream(reference_pdf):
    reference, ours = documents(reference_pdf[0])
    expected = reference[1].get_text("words", delimiters=".-")
    actual = ours[1].get_text("words", delimiters=".-")
    assert [item[4] for item in actual] == [item[4] for item in expected]


def test_sort_reorders_positioned_content():
    document = upstream.open()
    page = document.new_page(width=200, height=150)
    page.insert_text((20, 100), "lower")
    page.insert_text((20, 30), "upper")
    data = document.tobytes(deflate=True)
    reference, ours = documents(data)
    expected = reference[0].get_text("words", sort=True)
    actual = ours[0].get_text("words", sort=True)
    assert [word[4] for word in actual] == [word[4] for word in expected] == ["upper", "lower"]
    assert ours[0].get_text(sort=False) == reference[0].get_text(sort=False)


def test_unicode_type0_font_and_tounicode_map(unicode_pdf):
    reference, ours = documents(unicode_pdf)
    assert ours[0].get_text() == reference[0].get_text() == "Café Ω Привет\n"
    expected, actual = reference[0].get_text("words"), ours[0].get_text("words")
    assert [item[4] for item in actual] == [item[4] for item in expected]
    np.testing.assert_allclose(boxes(actual), boxes(expected), atol=5e-4)


def test_form_xobject_text_and_transform(form_pdf):
    reference, ours = documents(form_pdf)
    assert ours[0].get_text() == reference[0].get_text()
    np.testing.assert_allclose(
        boxes(ours[0].get_text("words")), boxes(reference[0].get_text("words")), atol=5e-4
    )


def test_compressed_object_stream_pdf_matches_upstream():
    document = upstream.open()
    page = document.new_page(width=240, height=100)
    page.insert_text((20, 35), "Objects may be compressed")
    data = document.tobytes(deflate=True, garbage=4, use_objstms=1)
    reference, ours = documents(data)
    assert ours[0].get_text() == reference[0].get_text()
    np.testing.assert_allclose(
        boxes(ours[0].get_text("words")), boxes(reference[0].get_text("words")), atol=5e-4
    )


def test_html_xhtml_and_xml_modes_are_available(reference_pdf):
    page = mupdf.open(stream=reference_pdf[0])[0]
    assert "<div" in page.get_text("html") and "Page 1" in page.get_text("html")
    assert page.get_text("xhtml").startswith("<div>")
    assert page.get_text("xml").startswith("<page>")


def test_search_for_returns_matching_rect(reference_pdf):
    page = mupdf.open(stream=reference_pdf[0])[0]
    matches = page.search_for("alpha")
    assert len(matches) == 1
    assert isinstance(matches[0], mupdf.Rect)


def test_context_manager_closes_document(reference_pdf):
    with mupdf.open(stream=reference_pdf[0]) as document:
        assert not document.is_closed
        assert len(document) == 2
    assert document.is_closed
    with pytest.raises(ValueError):
        len(document)


def test_invalid_and_empty_data_raise_compatible_errors():
    with pytest.raises(mupdf.EmptyFileError):
        mupdf.open(stream=b"")
    with pytest.raises(mupdf.FileDataError):
        mupdf.open(stream=b"not a pdf")
