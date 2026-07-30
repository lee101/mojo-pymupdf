"""PDF text-state interpreter fed by the Mojo content lexer."""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable

import numpy as np

from . import _lib
from ._pdf import Name, PDF, Ref


HELVETICA_WIDTHS = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]
TIMES_WIDTHS = [
    250, 333, 408, 500, 500, 833, 778, 180, 333, 333, 500, 564, 250, 333, 250, 278,
    500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 278, 278, 564, 564, 564, 444,
    921, 722, 667, 667, 722, 611, 556, 722, 722, 333, 389, 722, 611, 889, 722, 722,
    556, 722, 667, 556, 611, 722, 722, 944, 722, 722, 611, 333, 278, 333, 469, 500,
    333, 444, 500, 444, 500, 444, 333, 500, 500, 278, 278, 500, 278, 778, 500, 500,
    500, 500, 333, 389, 278, 500, 500, 722, 500, 500, 444, 480, 200, 480, 541,
]


@dataclass
class Font:
    name: str = "Helvetica"
    widths: dict[int, float] | None = None
    default_width: float = 500.0
    cmap: dict[int, str] | None = None
    code_bytes: int = 1
    ascender: float = 1.075
    descender: float = -0.299
    flags: int = 0

    def width(self, code: int) -> float:
        if self.widths and code in self.widths:
            return self.widths[code]
        if self.name.startswith("Courier"):
            return 600.0
        table = TIMES_WIDTHS if self.name.startswith("Times") else HELVETICA_WIDTHS
        return float(table[code - 32]) if 32 <= code <= 126 else self.default_width

    def decode(self, raw: bytes) -> list[tuple[str, int]]:
        if self.code_bytes == 2:
            codes = [int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw) - 1, 2)]
        else:
            codes = raw
            if not self.cmap:
                return list(zip(raw.decode("cp1252", "replace"), codes))
        result = []
        for code in codes:
            if self.cmap and code in self.cmap:
                text = self.cmap[code]
            elif self.code_bytes == 1:
                text = bytes([code]).decode("cp1252", "replace")
            else:
                text = chr(code) if code <= 0x10FFFF else "\ufffd"
            result.append((text, code))
        return result


def _font_name(value: Any) -> str:
    name = str(value or "/Helvetica").lstrip("/")
    name = re.sub(r"^[A-Z]{6}\+", "", name)
    aliases = {"Arial": "Helvetica", "ArialMT": "Helvetica", "CourierNew": "Courier"}
    return aliases.get(name, name)


def _to_unicode(pdf: PDF, reference: Any) -> dict[int, str]:
    if not isinstance(reference, Ref):
        return {}
    data = pdf.decoded_stream(reference)
    mapping: dict[int, str] = {}
    for section in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for source, target in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", section):
            if len(target) & 1:
                mapping[int(source, 16)] = chr(int(target, 16))
            else:
                raw = bytes.fromhex(target.decode())
                mapping[int(source, 16)] = raw.decode("utf-16-be", "replace")
    for section in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for line in section.splitlines():
            match = re.match(
                rb"\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
                line,
            )
            if match:
                first, last, base = (int(value, 16) for value in match.groups())
                for offset, code in enumerate(range(first, last + 1)):
                    value = base + offset
                    mapping[code] = chr(value) if value <= 0x10FFFF else "\ufffd"
                continue
            match = re.match(
                rb"\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*\[(.*?)\]",
                line,
            )
            if match:
                first, last = int(match.group(1), 16), int(match.group(2), 16)
                targets = re.findall(rb"<([0-9A-Fa-f]+)>", match.group(3))
                for code, target in zip(range(first, last + 1), targets):
                    mapping[code] = bytes.fromhex(target.decode()).decode("utf-16-be", "replace")
    return mapping


def load_fonts(pdf: PDF, resources_value: Any) -> tuple[dict[str, Font], dict[str, Any]]:
    resources = pdf.dictionary(resources_value)
    font_entries = pdf.dictionary(resources.get("/Font"))
    fonts: dict[str, Font] = {}
    for resource_name, reference in font_entries.items():
        dictionary = pdf.dictionary(reference)
        subtype = dictionary.get("/Subtype")
        descendant = {}
        if subtype == "/Type0":
            descendants = pdf.object(dictionary.get("/DescendantFonts", [])) or []
            descendant = pdf.dictionary(descendants[0]) if descendants else {}
        base = _font_name(descendant.get("/BaseFont") or dictionary.get("/BaseFont"))
        first = int(dictionary.get("/FirstChar", 0) or 0)
        widths_list = pdf.object(dictionary.get("/Widths", [])) or []
        widths = {first + i: float(width) for i, width in enumerate(widths_list)}
        default = float(descendant.get("/DW", 600 if base.startswith("Courier") else 500) or 500)
        if descendant.get("/W"):
            entries = pdf.object(descendant["/W"]) or []
            i = 0
            while i < len(entries):
                start = int(entries[i])
                next_value = pdf.object(entries[i + 1])
                if isinstance(next_value, list):
                    for offset, width in enumerate(next_value):
                        widths[start + offset] = float(width)
                    i += 2
                else:
                    end, width = int(next_value), float(entries[i + 2])
                    for code in range(start, end + 1):
                        widths[code] = width
                    i += 3
        descriptor = pdf.dictionary(descendant.get("/FontDescriptor") or dictionary.get("/FontDescriptor"))
        asc = float(descriptor.get("/Ascent", 1075)) / 1000
        desc = float(descriptor.get("/Descent", -299)) / 1000
        if base.startswith("Courier") and not descriptor:
            asc, desc = 0.932, -0.317
        elif base.startswith("Times") and not descriptor:
            asc, desc = 1.053, -0.281
        flags = 0
        lower = base.lower()
        if "italic" in lower or "oblique" in lower:
            flags |= 2
        if "times" in lower or "serif" in lower:
            flags |= 4
        if "courier" in lower or "mono" in lower:
            flags |= 8
        if "bold" in lower:
            flags |= 16
        fonts[resource_name.lstrip("/")] = Font(
            base,
            widths,
            default,
            _to_unicode(pdf, dictionary.get("/ToUnicode")),
            2 if subtype == "/Type0" else 1,
            asc,
            desc,
            flags,
        )
    xobjects = pdf.dictionary(resources.get("/XObject"))
    return fonts, xobjects


Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)


def multiply(left: Matrix, right: Matrix) -> Matrix:
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h,
        b * g + d * h,
        a * i + c * j,
        b * i + d * j,
        a * k + c * l + e,
        b * k + d * l + f,
    )


def transform(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


@dataclass
class Character:
    c: str
    origin: tuple[float, float]
    bbox: tuple[float, float, float, float]


@dataclass
class Run:
    chars: list[Character]
    font: Font
    size: float
    origin: tuple[float, float]
    direction: tuple[float, float] = (1.0, 0.0)

    @cached_property
    def text(self) -> str:
        return "".join(char.c for char in self.chars)

    @cached_property
    def bbox(self) -> tuple[float, float, float, float]:
        return union_boxes(char.bbox for char in self.chars)


def union_boxes(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    iterator = iter(boxes)
    try:
        x0, y0, x1, y1 = next(iterator)
    except StopIteration:
        return (0.0, 0.0, 0.0, 0.0)
    for box in iterator:
        x0 = min(x0, box[0])
        y0 = min(y0, box[1])
        x1 = max(x1, box[2])
        y1 = max(y1, box[3])
    return x0, y0, x1, y1


class Interpreter:
    def __init__(self, pdf: PDF, resources: Any, height: float):
        self.pdf = pdf
        self.height = height
        self.fonts, self.xobjects = load_fonts(pdf, resources)
        self.runs: list[Run] = []
        self.ctm: Matrix = IDENTITY
        self.graphics_stack: list[Matrix] = []
        self.tm: Matrix = IDENTITY
        self.line_matrix: Matrix = IDENTITY
        self.font = Font()
        self.font_size = 12.0
        self.char_space = 0.0
        self.word_space = 0.0
        self.hscale = 1.0
        self.leading = 0.0
        self.rise = 0.0

    def _page_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return point[0], self.height - point[1]

    def _show(self, raw: bytes) -> None:
        matrix = multiply(self.ctm, self.tm)
        decoded = self.font.decode(raw)
        if not decoded:
            return
        texts = [item[0] for item in decoded]
        advances = np.fromiter(
            (
                (
                    self.font.width(code) / 1000 * self.font_size
                    + self.char_space
                    + (self.word_space if text == " " else 0.0)
                )
                * self.hscale
                for text, code in decoded
            ),
            dtype=np.float64,
            count=len(decoded),
        )
        positions = np.empty_like(advances)
        positions[0] = 0.0
        if len(positions) > 1:
            np.cumsum(advances[:-1], out=positions[1:])
        geometry = _lib.layout_glyphs(
            positions,
            advances,
            matrix,
            self.rise + self.font.descender * self.font_size,
            self.rise + self.font.ascender * self.font_size,
            self.rise,
            self.height,
        )
        chars: list[Character] = []
        a, b, _, _, _, _ = matrix
        norm = math.hypot(a, b) or 1.0
        direction = (a / norm, -b / norm)
        for index, text in enumerate(texts):
            advance = advances[index]
            origin = (geometry[0, index], geometry[1, index])
            bbox = tuple(geometry[2:, index])
            if len(text) <= 1:
                chars.append(Character(text, origin, bbox))
            else:
                each = advance / len(text)
                for text_index, char in enumerate(text):
                    x0 = bbox[0] + text_index * each
                    x1 = bbox[0] + (text_index + 1) * each
                    chars.append(
                        Character(
                            char,
                            (origin[0] + text_index * each, origin[1]),
                            (x0, bbox[1], x1, bbox[3]),
                        )
                    )
        total = float(positions[-1] + advances[-1])
        a, b, c, d, e, f = self.tm
        self.tm = (a, b, c, d, a * total + e, b * total + f)
        self.runs.append(
            Run(chars, self.font, self.font_size, (geometry[0, 0], geometry[1, 0]), direction)
        )

    def _adjust(self, value: float) -> None:
        amount = -value / 1000 * self.font_size * self.hscale
        self.tm = multiply(self.tm, (1, 0, 0, 1, amount, 0))

    def process(self, data: bytes, resources: Any | None = None, depth: int = 0) -> None:
        if depth > 8 or not data:
            return
        if resources is not None:
            old_fonts, old_xobjects = self.fonts, self.xobjects
            self.fonts, self.xobjects = load_fonts(self.pdf, resources)
        else:
            old_fonts = old_xobjects = None
        kinds, offsets, lengths = _lib.lex(data)
        stack: list[Any] = []
        arrays: list[list[Any]] = []
        inline_image = False
        for kind, offset, length in zip(kinds, offsets, lengths):
            raw = data[int(offset) : int(offset + length)]
            if kind in (3, 4):
                value: Any = _lib.decode_string(raw, int(kind))
            elif kind == 2:
                value = Name("/" + raw.decode("latin1"))
            elif kind == 5:
                arrays.append([])
                continue
            elif kind == 6:
                value = arrays.pop() if arrays else []
            elif kind in (7, 8):
                continue
            else:
                word = raw.decode("latin1")
                try:
                    value = float(word) if any(c in word for c in ".eE") else int(word)
                except ValueError:
                    if inline_image:
                        if word == "EI":
                            inline_image = False
                        continue
                    if word == "BI":
                        inline_image = True
                        stack.clear()
                        continue
                    self.operator(word, stack, depth)
                    stack.clear()
                    continue
            if arrays:
                arrays[-1].append(value)
            else:
                stack.append(value)
        if resources is not None:
            self.fonts, self.xobjects = old_fonts, old_xobjects

    def operator(self, op: str, values: list[Any], depth: int) -> None:
        def nums(count: int) -> list[float]:
            return [float(value) for value in values[-count:]]

        if op == "q":
            self.graphics_stack.append(self.ctm)
        elif op == "Q":
            if self.graphics_stack:
                self.ctm = self.graphics_stack.pop()
        elif op == "cm" and len(values) >= 6:
            self.ctm = multiply(self.ctm, tuple(nums(6)))  # type: ignore[arg-type]
        elif op == "BT":
            self.tm = self.line_matrix = IDENTITY
        elif op == "Tf" and len(values) >= 2:
            self.font = self.fonts.get(str(values[-2]).lstrip("/"), Font(_font_name(values[-2])))
            self.font_size = float(values[-1])
        elif op == "Tm" and len(values) >= 6:
            self.tm = self.line_matrix = tuple(nums(6))  # type: ignore[assignment]
        elif op in ("Td", "TD") and len(values) >= 2:
            tx, ty = nums(2)
            if op == "TD":
                self.leading = -ty
            self.line_matrix = multiply(self.line_matrix, (1, 0, 0, 1, tx, ty))
            self.tm = self.line_matrix
        elif op == "T*":
            self.line_matrix = multiply(self.line_matrix, (1, 0, 0, 1, 0, -self.leading))
            self.tm = self.line_matrix
        elif op == "TL" and values:
            self.leading = float(values[-1])
        elif op == "Tc" and values:
            self.char_space = float(values[-1])
        elif op == "Tw" and values:
            self.word_space = float(values[-1])
        elif op == "Tz" and values:
            self.hscale = float(values[-1]) / 100
        elif op == "Ts" and values:
            self.rise = float(values[-1])
        elif op == "Tj" and values and isinstance(values[-1], bytes):
            self._show(values[-1])
        elif op == "TJ" and values and isinstance(values[-1], list):
            for item in values[-1]:
                self._show(item) if isinstance(item, bytes) else self._adjust(float(item))
        elif op in ("'", '"') and values:
            if op == '"' and len(values) >= 3:
                self.word_space, self.char_space = float(values[-3]), float(values[-2])
            self.operator("T*", [], depth)
            if isinstance(values[-1], bytes):
                self._show(values[-1])
        elif op == "Do" and values and isinstance(values[-1], Name):
            reference = self.xobjects.get(str(values[-1]))
            obj = self.pdf.indirect(reference)
            if obj and isinstance(obj.value, dict) and obj.value.get("/Subtype") == "/Form":
                saved = self.ctm
                matrix = self.pdf.object(obj.value.get("/Matrix", list(IDENTITY)))
                if isinstance(matrix, list) and len(matrix) == 6:
                    self.ctm = multiply(self.ctm, tuple(float(x) for x in matrix))  # type: ignore[arg-type]
                self.process(self.pdf.decoded_stream(obj), obj.value.get("/Resources"), depth + 1)
                self.ctm = saved


def _intersects(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class TextPage:
    def __init__(self, runs: list[Run], width: float, height: float, clip=None):
        if clip is not None:
            box = tuple(clip)
            runs = [
                Run(
                    [char for char in run.chars if _intersects(char.bbox, box)],
                    run.font,
                    run.size,
                    run.origin,
                    run.direction,
                )
                for run in runs
            ]
            runs = [run for run in runs if run.chars]
        self.width = width
        self.height = height
        self.runs = runs
        self._line_cache: dict[bool, list[list[Run]]] = {}
        self._block_cache: dict[bool, list[list[list[Run]]]] = {}
        self._text_cache: dict[bool, str] = {}
        self._word_cache: dict[str, tuple[tuple[Any, ...], ...]] = {}

    def _lines(self, sort: bool = False) -> list[list[Run]]:
        cached = self._line_cache.get(sort)
        if cached is not None:
            return cached
        lines: list[list[Run]] = []
        for run in self.runs:
            found = None
            for line in reversed(lines[-8:]):
                tolerance = max(1.0, run.size * 0.2)
                if abs(line[0].origin[1] - run.origin[1]) <= tolerance:
                    found = line
                    break
            if found is None:
                lines.append([run])
            else:
                found.append(run)
        for line in lines:
            line.sort(key=lambda item: item.bbox[0])
        if sort:
            lines.sort(
                key=lambda line: (
                    union_boxes(run.bbox for run in line)[1],
                    union_boxes(run.bbox for run in line)[0],
                )
            )
        self._line_cache[sort] = lines
        return lines

    def _blocks(self, sort: bool = False) -> list[list[list[Run]]]:
        cached = self._block_cache.get(sort)
        if cached is not None:
            return cached
        blocks: list[list[list[Run]]] = []
        for line in self._lines(sort):
            if not blocks:
                blocks.append([line])
                continue
            previous = blocks[-1][-1]
            size = max([run.size for run in previous + line] or [12.0])
            y_distance = abs(previous[0].origin[1] - line[0].origin[1])
            x_distance = abs(
                union_boxes(run.bbox for run in previous)[0]
                - union_boxes(run.bbox for run in line)[0]
            )
            same_direction = sum(
                a * b for a, b in zip(previous[0].direction, line[0].direction)
            ) > 0.98
            if same_direction and y_distance <= size * 1.6 and x_distance <= size * 2:
                blocks[-1].append(line)
            else:
                blocks.append([line])
        self._block_cache[sort] = blocks
        return blocks

    def extractText(self, sort: bool = False) -> str:
        cached = self._text_cache.get(sort)
        if cached is None:
            cached = "".join(
                "".join(run.text for run in line) + "\n" for line in self._lines(sort)
            )
            self._text_cache[sort] = cached
        return cached

    extractTEXT = extractText

    def extractWORDS(self, delimiters=None):
        cache_key = delimiters or ""
        cached = self._word_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        result = []
        delimiter_set = set(cache_key)
        for block_no, block in enumerate(self._blocks()):
            for line_no, line in enumerate(block):
                word_text: list[str] = []
                x0 = y0 = x1 = y1 = 0.0
                word_no = 0
                for run in line:
                    for char in run.chars:
                        if char.c.isspace() or char.c in delimiter_set:
                            if word_text:
                                result.append(
                                    (
                                        x0,
                                        y0,
                                        x1,
                                        y1,
                                        "".join(word_text),
                                        block_no,
                                        line_no,
                                        word_no,
                                    )
                                )
                                word_no += 1
                                word_text = []
                            continue
                        if word_text:
                            x0 = min(x0, char.bbox[0])
                            y0 = min(y0, char.bbox[1])
                            x1 = max(x1, char.bbox[2])
                            y1 = max(y1, char.bbox[3])
                        else:
                            x0, y0, x1, y1 = char.bbox
                        word_text.append(char.c)
                if word_text:
                    result.append(
                        (x0, y0, x1, y1, "".join(word_text), block_no, line_no, word_no)
                    )
        self._word_cache[cache_key] = tuple(result)
        return result

    def extractBLOCKS(self):
        result = []
        for number, block in enumerate(self._blocks()):
            box = union_boxes(run.bbox for line in block for run in line)
            text = "".join("".join(run.text for run in line) + "\n" for line in block)
            result.append((*box, text, number, 0))
        return result

    def _dict(self, raw: bool, sort: bool = False) -> dict[str, Any]:
        blocks = []
        for number, block_runs in enumerate(self._blocks(sort)):
            lines = []
            for line_runs in block_runs:
                spans = []
                for run in line_runs:
                    span: dict[str, Any] = {
                        "size": run.size,
                        "flags": run.font.flags,
                        "bidi": 0,
                        "char_flags": 16,
                        "font": run.font.name,
                        "color": 0,
                        "alpha": 255,
                        "ascender": run.font.ascender,
                        "descender": run.font.descender,
                    }
                    if raw:
                        span["chars"] = [
                            {"origin": char.origin, "bbox": char.bbox, "c": char.c, "synthetic": False}
                            for char in run.chars
                        ]
                    else:
                        span["text"] = run.text
                    span["origin"] = run.origin
                    span["bbox"] = run.bbox
                    spans.append(span)
                line_box = union_boxes(run.bbox for run in line_runs)
                lines.append(
                    {
                        "spans": spans,
                        "wmode": 0,
                        "dir": line_runs[0].direction,
                        "bbox": line_box,
                    }
                )
            block_box = union_boxes(line["bbox"] for line in lines)
            blocks.append(
                {"type": 0, "number": number, "flags": 0, "bbox": block_box, "lines": lines}
            )
        return {"width": self.width, "height": self.height, "blocks": blocks}

    def extractDICT(self, cb=None, sort: bool = False):
        return self._dict(False, sort)

    def extractRAWDICT(self, cb=None, sort: bool = False):
        return self._dict(True, sort)

    def extractJSON(self, cb=None, sort: bool = False) -> str:
        return json.dumps(self.extractDICT(cb, sort), indent=1)

    def extractRAWJSON(self, cb=None, sort: bool = False) -> str:
        return json.dumps(self.extractRAWDICT(cb, sort), indent=1)

    def extractHTML(self) -> str:
        parts = [f'<div id="page0" style="width:{self.width}pt;height:{self.height}pt">']
        for line in self._lines():
            box = union_boxes(run.bbox for run in line)
            text = html.escape("".join(run.text for run in line))
            parts.append(f'<p style="top:{box[1]}pt;left:{box[0]}pt">{text}</p>')
        return "".join(parts) + "</div>"

    def extractXHTML(self) -> str:
        return "<div>" + "".join(f"<p>{html.escape(''.join(r.text for r in line))}</p>" for line in self._lines()) + "</div>"

    def extractXML(self) -> str:
        chars = "".join(
            f'<char quad="{c.bbox[0]} {c.bbox[1]} {c.bbox[2]} {c.bbox[3]}" c="{html.escape(c.c)}"/>'
            for run in self.runs for c in run.chars
        )
        return f"<page>{chars}</page>"
