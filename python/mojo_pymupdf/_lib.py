"""ctypes bridge to the Mojo PDF content kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIB = os.path.join(ROOT, "dist", "libmojo-pymupdf.so")
LIB = os.environ.get("MOJO_PYMUPDF_LIB", DEFAULT_LIB)
I = ctypes.c_int64
D = ctypes.c_double
LAYOUT_PARALLEL_THRESHOLD = 262_144

_handle: ctypes.CDLL | None = None
_runtime_handle: ctypes.CDLL | None = None
_cpu_device: int | None = None


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "pymupdf.mojo")
    if LIB != DEFAULT_LIB:
        if not os.path.isfile(LIB):
            raise RuntimeError(f"MOJO_PYMUPDF_LIB does not exist: {LIB}")
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(build())
        _handle.mpdf_lex.argtypes = [I, I, I, I, I, I]
        _handle.mpdf_lex.restype = I
        _handle.mpdf_decode_string.argtypes = [I, I, I, I]
        _handle.mpdf_decode_string.restype = I
        _handle.mpdf_layout_glyphs.argtypes = [I, I, I, I] + [D] * 10
        _handle.mpdf_layout_glyphs.restype = I
    return _handle


def _ensure_cpu_parallel_runtime() -> None:
    global _runtime_handle, _cpu_device
    if _cpu_device is None:
        lib()
        _runtime_handle = ctypes.CDLL("libKGENCompilerRTShared.so")
        create = _runtime_handle.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
        create.restype = ctypes.c_void_p
        _cpu_device = create()
        if not _cpu_device:
            raise RuntimeError("failed to initialize Mojo CPU runtime")


def lex(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    source = np.frombuffer(data, dtype=np.uint8)
    capacity = len(data) + 1
    kinds = np.empty(capacity, dtype=np.uint8)
    offsets = np.empty(capacity, dtype=np.int64)
    lengths = np.empty(capacity, dtype=np.int64)
    count = lib().mpdf_lex(
        source.ctypes.data,
        len(source),
        kinds.ctypes.data,
        offsets.ctypes.data,
        lengths.ctypes.data,
        capacity,
    )
    if count < 0:
        raise RuntimeError("Mojo lexer rejected its buffer contract")
    return kinds[:count], offsets[:count], lengths[:count]


def decode_string(raw: bytes, kind: int) -> bytes:
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if kind not in (3, 4):
        raise ValueError("kind must be 3 (literal) or 4 (hex)")
    if not raw:
        return b""
    source = np.frombuffer(raw, dtype=np.uint8)
    dest = np.empty(len(raw) + 1, dtype=np.uint8)
    size = lib().mpdf_decode_string(source.ctypes.data, len(source), kind, dest.ctypes.data)
    if not 0 <= size <= len(raw):
        raise RuntimeError("Mojo decoder rejected its buffer contract")
    return dest[:size].tobytes()


def layout_glyphs(
    positions: np.ndarray,
    advances: np.ndarray,
    matrix: tuple[float, float, float, float, float, float],
    low: float,
    high: float,
    rise: float,
    page_height: float,
) -> np.ndarray:
    original_positions = np.asarray(positions)
    original_advances = np.asarray(advances)
    for name, value in (
        ("positions", original_positions),
        ("advances", original_advances),
    ):
        if value.dtype.kind not in "fiu":
            raise TypeError(f"{name} must contain real numeric values")
        if value.dtype.kind == "f" and value.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError(f"{name} cannot be narrowed to float64")
        if value.dtype.kind in "iu" and value.size:
            too_large = np.any(value > 2**53)
            too_small = value.dtype.kind == "i" and np.any(value < -(2**53))
            if too_large or too_small:
                raise TypeError(
                    f"{name} contains integers not exactly representable as float64"
                )
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    advances = np.ascontiguousarray(advances, dtype=np.float64)
    if positions.ndim != 1 or advances.shape != positions.shape:
        raise ValueError("positions and advances must be equal-length vectors")
    if len(matrix) != 6:
        raise ValueError("matrix must contain exactly six values")
    matrix = tuple(float(value) for value in matrix)
    geometry = np.empty((6, len(positions)), dtype=np.float64)
    if not len(positions):
        return geometry
    if len(positions) >= LAYOUT_PARALLEL_THRESHOLD:
        _ensure_cpu_parallel_runtime()
    status = lib().mpdf_layout_glyphs(
        positions.ctypes.data,
        advances.ctypes.data,
        geometry.ctypes.data,
        len(positions),
        *matrix,
        low,
        high,
        rise,
        page_height,
    )
    if status != 0:
        raise RuntimeError("Mojo layout kernel rejected its buffer contract")
    return geometry
