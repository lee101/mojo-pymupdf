"""A Mojo-accelerated subset of PyMuPDF for PDF page text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from ._pdf import EmptyFileError, FileDataError, PDF, Ref, read_source
from ._text import Interpreter, TextPage

__version__ = "0.1.0"
version = (__version__, "Mojo PDF text core", None)

TEXT_PRESERVE_LIGATURES = 1
TEXT_PRESERVE_WHITESPACE = 2
TEXT_PRESERVE_IMAGES = 4
TEXT_INHIBIT_SPACES = 8
TEXT_DEHYPHENATE = 16
TEXT_PRESERVE_SPANS = 32
TEXT_MEDIABOX_CLIP = 64
TEXT_CID_FOR_UNKNOWN_UNICODE = 128
TEXTFLAGS_TEXT = 195
TEXTFLAGS_WORDS = 195
TEXTFLAGS_BLOCKS = 195
TEXTFLAGS_DICT = 199
TEXTFLAGS_RAWDICT = 199
TEXTFLAGS_HTML = 199
TEXTFLAGS_XHTML = 199
TEXTFLAGS_XML = 195


@dataclass(frozen=True)
class Point:
    x: float = 0.0
    y: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass(frozen=True)
class Rect:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def __init__(self, *args):
        if len(args) == 1:
            args = tuple(args[0])
        if len(args) == 2:
            args = (*args[0], *args[1])
        if len(args) != 4:
            raise ValueError("Rect: bad args")
        object.__setattr__(self, "x0", float(args[0]))
        object.__setattr__(self, "y0", float(args[1]))
        object.__setattr__(self, "x1", float(args[2]))
        object.__setattr__(self, "y1", float(args[3]))

    def __iter__(self):
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    @property
    def width(self):
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self):
        return max(0.0, self.y1 - self.y0)

    @property
    def is_empty(self):
        return self.width == 0 or self.height == 0

    def intersects(self, other) -> bool:
        other = Rect(other)
        return self.x0 < other.x1 and self.x1 > other.x0 and self.y0 < other.y1 and self.y1 > other.y0


class Page:
    def __init__(self, parent: "Document", number: int, reference: Ref):
        self.parent = parent
        self.number = number
        self._reference = reference

    @property
    def rect(self) -> Rect:
        box = self.parent._pdf.object(
            self.parent._pdf.inherited(self._reference, "/CropBox")
            or self.parent._pdf.inherited(self._reference, "/MediaBox", [0, 0, 612, 792])
        )
        if not isinstance(box, list) or len(box) != 4:
            box = [0, 0, 612, 792]
        return Rect(0, 0, float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))

    mediabox = rect

    @property
    def rotation(self) -> int:
        return int(self.parent._pdf.object(self.parent._pdf.inherited(self._reference, "/Rotate", 0)) or 0)

    def bound(self) -> Rect:
        return self.rect

    def get_textpage(self, clip=None, flags: int = 0, matrix=None) -> TextPage:
        self.parent._ensure_open()
        page = self.parent._textpages.get(self.number)
        if page is None:
            resources = self.parent._pdf.inherited(self._reference, "/Resources", {})
            interpreter = Interpreter(self.parent._pdf, resources, self.rect.height)
            interpreter.process(self.parent._pdf.page_stream(self._reference))
            page = TextPage(interpreter.runs, self.rect.width, self.rect.height)
            self.parent._textpages[self.number] = page
        if clip is not None:
            return TextPage(page.runs, page.width, page.height, clip)
        return page

    def get_text(
        self,
        option: str = "text",
        clip=None,
        flags=None,
        textpage: TextPage | None = None,
        sort: bool = False,
        delimiters=None,
    ):
        page = textpage or self.get_textpage(clip=clip, flags=flags or 0)
        option = option.lower()
        if option in ("text", ""):
            return page.extractText(sort=sort)
        if option == "words":
            words = page.extractWORDS(delimiters=delimiters)
            return sorted(words, key=lambda word: (word[1], word[0])) if sort else words
        if option == "blocks":
            blocks = page.extractBLOCKS()
            return sorted(blocks, key=lambda block: (block[1], block[0])) if sort else blocks
        if option == "dict":
            return page.extractDICT(sort=sort)
        if option == "rawdict":
            return page.extractRAWDICT(sort=sort)
        if option == "json":
            return page.extractJSON(sort=sort)
        if option == "rawjson":
            return page.extractRAWJSON(sort=sort)
        if option == "html":
            return page.extractHTML()
        if option == "xhtml":
            return page.extractXHTML()
        if option == "xml":
            return page.extractXML()
        raise ValueError(f"unknown output option: {option}")

    getText = get_text

    def search_for(self, needle: str, clip=None, quads: bool = False, flags: int = TEXTFLAGS_TEXT, textpage=None):
        if quads:
            raise NotImplementedError("quad output is outside the covered subset")
        page = textpage or self.get_textpage(clip=clip, flags=flags)
        matches = []
        lower = needle.casefold()
        for block in page.extractBLOCKS():
            if lower in block[4].casefold():
                matches.append(Rect(block[:4]))
        return matches


class Document:
    def __init__(self, data: bytes, name: str | None = None, pdf: PDF | None = None):
        self._pdf = pdf or PDF(data)
        self.name = name
        self.is_closed = False
        self.is_pdf = True
        self.needs_pass = False
        self.is_encrypted = b"/Encrypt" in data[-4096:]
        self._textpages: dict[int, TextPage] = {}

    def _ensure_open(self):
        if self.is_closed:
            raise ValueError("document closed")

    @property
    def page_count(self) -> int:
        self._ensure_open()
        return len(self._pdf.pages)

    def __len__(self) -> int:
        return self.page_count

    def load_page(self, page_id: int = 0) -> Page:
        self._ensure_open()
        if page_id < 0:
            page_id += len(self._pdf.pages)
        if not 0 <= page_id < len(self._pdf.pages):
            raise ValueError("page not in document")
        return Page(self, page_id, self._pdf.pages[page_id])

    loadPage = load_page

    def __getitem__(self, page_id: int) -> Page:
        return self.load_page(page_id)

    def __iter__(self) -> Iterator[Page]:
        for number in range(len(self)):
            yield self.load_page(number)

    def get_page_text(self, pno: int, *args, **kwargs):
        return self.load_page(pno).get_text(*args, **kwargs)

    def close(self) -> None:
        self.is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def open(
    filename=None,
    stream=None,
    filetype=None,
    rect=None,
    width=0,
    height=0,
    fontsize=11,
    archive=None,
) -> Document:
    if filename is None and stream is None:
        raise ValueError("mojo-pymupdf only opens existing PDF data")
    data, name = read_source(filename, stream)
    return Document(data, name, _cached_pdf(data))


@lru_cache(maxsize=8)
def _cached_pdf(data: bytes) -> PDF:
    return PDF(data)


__all__ = [
    "Document",
    "EmptyFileError",
    "FileDataError",
    "Page",
    "Point",
    "Rect",
    "TextPage",
    "open",
]
