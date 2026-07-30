"""Small standalone PDF container parser for page and resource discovery."""

from __future__ import annotations

import base64
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_OBJECT_RE = re.compile(rb"(?m)(?<!\d)(\d+)\s+(\d+)\s+obj\b")
_STREAM_RE = re.compile(rb"\bstream(?:\r\n|\n|\r)")
_NAME_ESCAPE_RE = re.compile(rb"#([0-9A-Fa-f]{2})")
_LITERAL_ESCAPES = {110: 10, 114: 13, 116: 9, 98: 8, 102: 12}


class FileDataError(RuntimeError):
    pass


class EmptyFileError(FileDataError):
    pass


class Name(str):
    pass


@dataclass(frozen=True)
class Ref:
    number: int
    generation: int = 0


@dataclass
class IndirectObject:
    value: Any
    stream: bytes | None = None


def _decode_literal(raw: bytes) -> bytes:
    result = bytearray()
    i = 0
    depth = 1
    while i < len(raw):
        c = raw[i]
        i += 1
        if c == 92:
            if i >= len(raw):
                break
            c = raw[i]
            i += 1
            if c in _LITERAL_ESCAPES:
                result.append(_LITERAL_ESCAPES[c])
            elif c in (10, 13):
                if c == 13 and i < len(raw) and raw[i] == 10:
                    i += 1
            elif 48 <= c <= 55:
                digits = bytes([c])
                while len(digits) < 3 and i < len(raw) and 48 <= raw[i] <= 55:
                    digits += bytes([raw[i]])
                    i += 1
                result.append(int(digits, 8) & 255)
            else:
                result.append(c)
        else:
            if c == 40:
                depth += 1
            elif c == 41:
                depth -= 1
                if depth == 0:
                    break
            result.append(c)
    return bytes(result)


def _tokens(data: bytes) -> list[Any]:
    result: list[Any] = []
    i = 0
    n = len(data)
    white = b"\x00\t\n\f\r "
    delimiters = b"()<>[]{}/%"
    while i < n:
        if data[i] in white:
            i += 1
            continue
        if data[i] == 37:
            while i < n and data[i] not in b"\r\n":
                i += 1
            continue
        if data.startswith(b"<<", i):
            result.append("<<")
            i += 2
            continue
        if data.startswith(b">>", i):
            result.append(">>")
            i += 2
            continue
        if data[i] in b"[]":
            result.append(chr(data[i]))
            i += 1
            continue
        if data[i] == 47:
            i += 1
            start = i
            while i < n and data[i] not in white + delimiters:
                i += 1
            raw = data[start:i]
            if b"#" in raw:
                raw = _NAME_ESCAPE_RE.sub(
                    lambda match: bytes([int(match.group(1), 16)]), raw
                )
            result.append(Name("/" + raw.decode("latin1")))
            continue
        if data[i] == 40:
            start = i
            i += 1
            depth = 1
            escaped = False
            while i < n and depth:
                c = data[i]
                if escaped:
                    escaped = False
                elif c == 92:
                    escaped = True
                elif c == 40:
                    depth += 1
                elif c == 41:
                    depth -= 1
                i += 1
            result.append(_decode_literal(data[start + 1 : i]))
            continue
        if data[i] == 60:
            end = data.find(b">", i + 1)
            if end < 0:
                end = n
            raw = re.sub(rb"\s+", b"", data[i + 1 : end])
            if len(raw) & 1:
                raw += b"0"
            try:
                result.append(bytes.fromhex(raw.decode("ascii")))
            except ValueError:
                result.append(b"")
            i = min(end + 1, n)
            continue
        if data[i] in delimiters:
            result.append(chr(data[i]))
            i += 1
            continue
        start = i
        while i < n and data[i] not in white + delimiters:
            i += 1
        raw_bytes = data[start:i]
        position = 1 if raw_bytes[:1] in (b"+", b"-") else 0
        digit_start = position
        while position < len(raw_bytes) and 48 <= raw_bytes[position] <= 57:
            position += 1
        if position == len(raw_bytes) and position > digit_start:
            result.append(int(raw_bytes))
        elif position < len(raw_bytes) and raw_bytes[position] == 46:
            position += 1
            fractional_start = position
            while position < len(raw_bytes) and 48 <= raw_bytes[position] <= 57:
                position += 1
            if (
                position == len(raw_bytes)
                and (digit_start < fractional_start - 1 or fractional_start < position)
            ):
                result.append(float(raw_bytes))
            else:
                result.append(raw_bytes.decode("latin1"))
        else:
            result.append(raw_bytes.decode("latin1"))
    return result


def parse_value(data: bytes) -> Any:
    tokens = _tokens(data)
    position = 0

    def take() -> Any:
        nonlocal position
        if position >= len(tokens):
            return None
        token = tokens[position]
        position += 1
        if token == "<<":
            value = {}
            while position < len(tokens) and tokens[position] != ">>":
                key = take()
                item = take()
                if isinstance(key, Name):
                    value[str(key)] = item
            if position < len(tokens):
                position += 1
            return value
        if token == "[":
            value = []
            while position < len(tokens) and tokens[position] != "]":
                value.append(take())
            if position < len(tokens):
                position += 1
            return value
        if isinstance(token, int) and position + 1 < len(tokens):
            if isinstance(tokens[position], int) and tokens[position + 1] == "R":
                reference = Ref(token, tokens[position])
                position += 2
                return reference
        if token == "true":
            return True
        if token == "false":
            return False
        if token == "null":
            return None
        return token

    return take()


class PDF:
    def __init__(self, data: bytes):
        if not data:
            raise EmptyFileError("Cannot open empty stream.")
        if not data.lstrip().startswith(b"%PDF-"):
            raise FileDataError("Failed to open stream: not a PDF.")
        self.data = data
        self.objects: dict[int, IndirectObject] = {}
        self._parse_objects()
        self._parse_object_streams()
        self.pages = self._find_pages()

    def _parse_objects(self) -> None:
        matches = list(_OBJECT_RE.finditer(self.data))
        for index, match in enumerate(matches):
            number = int(match.group(1))
            limit = matches[index + 1].start() if index + 1 < len(matches) else len(self.data)
            end = self.data.rfind(b"endobj", match.end(), limit)
            if end < 0:
                end = limit
            body = self.data[match.end() : end].strip()
            stream_match = _STREAM_RE.search(body)
            if stream_match:
                dictionary = parse_value(body[: stream_match.start()].strip())
                start = stream_match.end()
                length = dictionary.get("/Length") if isinstance(dictionary, dict) else None
                if isinstance(length, int) and start + length <= len(body):
                    stream = body[start : start + length]
                else:
                    stream_end = body.rfind(b"endstream")
                    stream = body[start:stream_end].rstrip(b"\r\n")
                self.objects[number] = IndirectObject(dictionary, stream)
            else:
                self.objects[number] = IndirectObject(parse_value(body))

    def _parse_object_streams(self) -> None:
        containers = [
            obj
            for obj in list(self.objects.values())
            if isinstance(obj.value, dict) and obj.value.get("/Type") == "/ObjStm"
        ]
        for container in containers:
            count = int(container.value.get("/N", 0) or 0)
            first = int(container.value.get("/First", 0) or 0)
            data = self.decoded_stream(container)
            header = [int(value) for value in re.findall(rb"\d+", data[:first])]
            pairs = list(zip(header[0::2], header[1::2]))[:count]
            for index, (number, offset) in enumerate(pairs):
                end = first + pairs[index + 1][1] if index + 1 < len(pairs) else len(data)
                body = data[first + offset : end].strip()
                if number not in self.objects:
                    self.objects[number] = IndirectObject(parse_value(body))

    def object(self, value: Any) -> Any:
        if isinstance(value, Ref):
            obj = self.objects.get(value.number)
            return obj.value if obj else None
        return value

    def indirect(self, value: Any) -> IndirectObject | None:
        if isinstance(value, Ref):
            return self.objects.get(value.number)
        return None

    def dictionary(self, value: Any) -> dict[str, Any]:
        resolved = self.object(value)
        return resolved if isinstance(resolved, dict) else {}

    def decoded_stream(self, value: Ref | IndirectObject) -> bytes:
        obj = self.indirect(value) if isinstance(value, Ref) else value
        if obj is None or obj.stream is None:
            return b""
        stream = obj.stream
        filters = obj.value.get("/Filter") if isinstance(obj.value, dict) else None
        filters = self.object(filters)
        if not isinstance(filters, list):
            filters = [filters] if filters else []
        for item in filters:
            name = str(self.object(item))
            if name in ("/FlateDecode", "/Fl"):
                stream = zlib.decompress(stream)
            elif name in ("/ASCIIHexDecode", "/AHx"):
                stream = bytes.fromhex(re.sub(rb"\s+|>", b"", stream).decode("ascii"))
            elif name in ("/ASCII85Decode", "/A85"):
                stream = base64.a85decode(stream, adobe=b"<~" in stream)
            elif name in ("/RunLengthDecode", "/RL"):
                decoded = bytearray()
                position = 0
                while position < len(stream):
                    length = stream[position]
                    position += 1
                    if length == 128:
                        break
                    if length < 128:
                        decoded.extend(stream[position : position + length + 1])
                        position += length + 1
                    elif position < len(stream):
                        decoded.extend(stream[position : position + 1] * (257 - length))
                        position += 1
                stream = bytes(decoded)
            else:
                raise FileDataError(f"Unsupported PDF stream filter: {name}")
        return stream

    def _find_pages(self) -> list[Ref]:
        catalog = next(
            (
                Ref(number)
                for number, obj in self.objects.items()
                if isinstance(obj.value, dict) and obj.value.get("/Type") == "/Catalog"
            ),
            None,
        )
        pages: list[Ref] = []

        def walk(reference: Ref) -> None:
            dictionary = self.dictionary(reference)
            if dictionary.get("/Type") == "/Page":
                pages.append(reference)
                return
            for kid in self.object(dictionary.get("/Kids", [])) or []:
                if isinstance(kid, Ref):
                    walk(kid)

        if catalog:
            root = self.dictionary(catalog).get("/Pages")
            if isinstance(root, Ref):
                walk(root)
        if not pages:
            pages = [
                Ref(number)
                for number, obj in sorted(self.objects.items())
                if isinstance(obj.value, dict) and obj.value.get("/Type") == "/Page"
            ]
        return pages

    def inherited(self, page: Ref, key: str, default: Any = None) -> Any:
        current: Any = page
        seen: set[int] = set()
        while isinstance(current, Ref) and current.number not in seen:
            seen.add(current.number)
            dictionary = self.dictionary(current)
            if key in dictionary:
                return dictionary[key]
            current = dictionary.get("/Parent")
        return default

    def page_stream(self, page: Ref) -> bytes:
        contents = self.dictionary(page).get("/Contents")
        contents = self.object(contents)
        if isinstance(contents, list):
            chunks = [self.decoded_stream(item) for item in contents if isinstance(item, Ref)]
            return b"\n".join(chunks)
        if isinstance(self.dictionary(page).get("/Contents"), Ref):
            return self.decoded_stream(self.dictionary(page)["/Contents"])
        return b""


def read_source(filename=None, stream=None) -> tuple[bytes, str | None]:
    if stream is not None:
        if hasattr(stream, "read"):
            stream = stream.read()
        if isinstance(stream, str):
            stream = stream.encode("latin1")
        return bytes(stream), None
    if filename is None:
        raise TypeError("bad filename")
    if hasattr(filename, "read"):
        return bytes(filename.read()), getattr(filename, "name", None)
    path = Path(filename)
    return path.read_bytes(), str(path)
