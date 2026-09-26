"""Hot loops for PDF content stream tokenization and string decoding."""

from std.sys import simd_width_of as simdwidthof

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def is_white(c: UInt8) -> Bool:
    return c == 0 or c == 9 or c == 10 or c == 12 or c == 13 or c == 32


def is_delim(c: UInt8) -> Bool:
    return (
        c == 40
        or c == 41
        or c == 60
        or c == 62
        or c == 91
        or c == 93
        or c == 123
        or c == 125
        or c == 47
        or c == 37
    )


def put_token(
    kinds: BPtr,
    offsets: IPtr,
    lengths: IPtr,
    capacity: Int,
    count: Int,
    kind: UInt8,
    start: Int,
    length: Int,
):
    if count < capacity:
        kinds[unsafe_offset=count] = kind
        offsets[unsafe_offset=count] = Int64(start)
        lengths[unsafe_offset=count] = Int64(length)


@export("mpdf_lex")
def mpdf_lex(
    src_addr: Int,
    n: Int,
    kinds_addr: Int,
    offsets_addr: Int,
    lengths_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if n < 0 or capacity < 0:
        return -1
    if n > 0 and (
        src_addr == 0
        or kinds_addr == 0
        or offsets_addr == 0
        or lengths_addr == 0
    ):
        return -1
    if n == 0:
        return 0
    var src = BPtr(unsafe_from_address=src_addr)
    var kinds = BPtr(unsafe_from_address=kinds_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var lengths = IPtr(unsafe_from_address=lengths_addr)
    var i = 0
    var count = 0
    while i < n:
        var c = src[unsafe_offset=i]
        if is_white(c):
            i += 1
            continue
        if c == 37:
            i += 1
            while (
                i < n
                and src[unsafe_offset=i] != 10
                and src[unsafe_offset=i] != 13
            ):
                i += 1
            continue
        if c == 40:
            var start = i + 1
            i += 1
            var depth = 1
            while i < n and depth > 0:
                c = src[unsafe_offset=i]
                if c == 92:
                    i += 1
                    if i < n:
                        if (
                            src[unsafe_offset=i] == 13
                            and i + 1 < n
                            and src[unsafe_offset=i + 1] == 10
                        ):
                            i += 2
                        else:
                            i += 1
                    continue
                if c == 40:
                    depth += 1
                elif c == 41:
                    depth -= 1
                i += 1
            var size = i - start - 1 if depth == 0 else i - start
            put_token(kinds, offsets, lengths, capacity, count, 3, start, size)
            count += 1
            continue
        if c == 60:
            if i + 1 < n and src[unsafe_offset=i + 1] == 60:
                put_token(kinds, offsets, lengths, capacity, count, 7, i, 2)
                count += 1
                i += 2
                continue
            var start = i + 1
            i += 1
            while i < n and src[unsafe_offset=i] != 62:
                i += 1
            put_token(
                kinds, offsets, lengths, capacity, count, 4, start, i - start
            )
            count += 1
            if i < n:
                i += 1
            continue
        if c == 62 and i + 1 < n and src[unsafe_offset=i + 1] == 62:
            put_token(kinds, offsets, lengths, capacity, count, 8, i, 2)
            count += 1
            i += 2
            continue
        if c == 91:
            put_token(kinds, offsets, lengths, capacity, count, 5, i, 1)
            count += 1
            i += 1
            continue
        if c == 93:
            put_token(kinds, offsets, lengths, capacity, count, 6, i, 1)
            count += 1
            i += 1
            continue
        var kind = UInt8(1)
        var start = i
        if c == 47:
            kind = 2
            start = i + 1
            i += 1
        while (
            i < n
            and not is_white(src[unsafe_offset=i])
            and not is_delim(src[unsafe_offset=i])
        ):
            i += 1
        if i == start and kind == 1:
            i += 1
        put_token(
            kinds, offsets, lengths, capacity, count, kind, start, i - start
        )
        count += 1
    return -count if count > capacity else count


def hex_value(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c - 48)
    if c >= 65 and c <= 70:
        return Int(c - 65) + 10
    if c >= 97 and c <= 102:
        return Int(c - 97) + 10
    return -1


@export("mpdf_decode_string")
def mpdf_decode_string(
    src_addr: Int,
    n: Int,
    kind: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if n < 0 or (kind != 3 and kind != 4):
        return -1
    if n > 0 and (src_addr == 0 or dst_addr == 0):
        return -1
    if n == 0:
        return 0
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var i = 0
    var written = 0
    if kind == 4:
        var high = -1
        while i < n:
            var v = hex_value(src[unsafe_offset=i])
            i += 1
            if v < 0:
                continue
            if high < 0:
                high = v
            else:
                dst[unsafe_offset=written] = UInt8((high << 4) | v)
                written += 1
                high = -1
        if high >= 0:
            dst[unsafe_offset=written] = UInt8(high << 4)
            written += 1
        return written
    while i < n:
        var c = src[unsafe_offset=i]
        i += 1
        if c != 92:
            dst[unsafe_offset=written] = c
            written += 1
            continue
        if i >= n:
            break
        c = src[unsafe_offset=i]
        i += 1
        if c == 10:
            continue
        if c == 13:
            if i < n and src[unsafe_offset=i] == 10:
                i += 1
            continue
        if c == 110:
            c = 10
        elif c == 114:
            c = 13
        elif c == 116:
            c = 9
        elif c == 98:
            c = 8
        elif c == 102:
            c = 12
        elif c >= 48 and c <= 55:
            var v = Int(c - 48)
            var digits = 1
            while (
                digits < 3
                and i < n
                and src[unsafe_offset=i] >= 48
                and src[unsafe_offset=i] <= 55
            ):
                v = (v << 3) | Int(src[unsafe_offset=i] - 48)
                i += 1
                digits += 1
            c = UInt8(v & 255)
        dst[unsafe_offset=written] = c
        written += 1
    return written


def layout_range(
    positions: FPtr,
    advances: FPtr,
    geometry: FPtr,
    n: Int,
    start: Int,
    end: Int,
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
    e: Float64,
    f: Float64,
    low: Float64,
    high: Float64,
    rise: Float64,
    page_height: Float64,
):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    var vector_end = end - (end - start) % W
    while i < vector_end:
        var x = positions.unsafe_load[width=W](i)
        var advance = advances.unsafe_load[width=W](i)
        var x1 = x + advance
        var origin_x = a * x + c * rise + e
        var origin_y = page_height - (b * x + d * rise + f)
        var low_x0 = a * x + c * low + e
        var low_x1 = a * x1 + c * low + e
        var high_x0 = a * x + c * high + e
        var high_x1 = a * x1 + c * high + e
        var low_y0 = page_height - (b * x + d * low + f)
        var low_y1 = page_height - (b * x1 + d * low + f)
        var high_y0 = page_height - (b * x + d * high + f)
        var high_y1 = page_height - (b * x1 + d * high + f)
        geometry.unsafe_store(i, origin_x)
        geometry.unsafe_store(n + i, origin_y)
        geometry.unsafe_store(
            2 * n + i, min(min(low_x0, low_x1), min(high_x0, high_x1))
        )
        geometry.unsafe_store(
            3 * n + i, min(min(low_y0, low_y1), min(high_y0, high_y1))
        )
        geometry.unsafe_store(
            4 * n + i, max(max(low_x0, low_x1), max(high_x0, high_x1))
        )
        geometry.unsafe_store(
            5 * n + i, max(max(low_y0, low_y1), max(high_y0, high_y1))
        )
        i += W
    while i < end:
        var x = positions[unsafe_offset=i]
        var x1 = x + advances[unsafe_offset=i]
        var origin_x = a * x + c * rise + e
        var origin_y = page_height - (b * x + d * rise + f)
        var low_x0 = a * x + c * low + e
        var low_x1 = a * x1 + c * low + e
        var high_x0 = a * x + c * high + e
        var high_x1 = a * x1 + c * high + e
        var low_y0 = page_height - (b * x + d * low + f)
        var low_y1 = page_height - (b * x1 + d * low + f)
        var high_y0 = page_height - (b * x + d * high + f)
        var high_y1 = page_height - (b * x1 + d * high + f)
        geometry[unsafe_offset=i] = origin_x
        geometry[unsafe_offset=n + i] = origin_y
        geometry[unsafe_offset=2 * n + i] = min(
            min(low_x0, low_x1), min(high_x0, high_x1)
        )
        geometry[unsafe_offset=3 * n + i] = min(
            min(low_y0, low_y1), min(high_y0, high_y1)
        )
        geometry[unsafe_offset=4 * n + i] = max(
            max(low_x0, low_x1), max(high_x0, high_x1)
        )
        geometry[unsafe_offset=5 * n + i] = max(
            max(low_y0, low_y1), max(high_y0, high_y1)
        )
        i += 1


@export("mpdf_layout_glyphs")
def mpdf_layout_glyphs(
    positions_addr: Int,
    advances_addr: Int,
    geometry_addr: Int,
    n: Int,
    a: Float64,
    b: Float64,
    c: Float64,
    d: Float64,
    e: Float64,
    f: Float64,
    low: Float64,
    high: Float64,
    rise: Float64,
    page_height: Float64,
) abi("C") -> Int:
    if n < 0:
        return -1
    if n > 0 and (
        positions_addr == 0 or advances_addr == 0 or geometry_addr == 0
    ):
        return -1
    if n == 0:
        return 0
    var positions = FPtr(unsafe_from_address=positions_addr)
    var advances = FPtr(unsafe_from_address=advances_addr)
    var geometry = FPtr(unsafe_from_address=geometry_addr)
    layout_range(
        positions,
        advances,
        geometry,
        n,
        0,
        n,
        a,
        b,
        c,
        d,
        e,
        f,
        low,
        high,
        rise,
        page_height,
    )
    return 0
