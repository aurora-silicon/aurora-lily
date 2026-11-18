#!/usr/bin/env python3
"""Converts Agnes 3.0 Flash (`Agnes-AI/Agnes-3.0-Flash`, the open-weight
Preview checkpoint, BF16 safetensors) into two GGUF files of llama.cpp's
`qwen35` architecture: the trunk and, beside it, the MTP block
(`mtp-*.gguf`), the layout lily's Qwen3.6 loader reads.

    .venv/bin/python tools/convert/convert_agnes3_flash.py \\
        --src ~/models/Agnes-3.0-Flash --dst ~/models/Agnes-3.0-Flash-gguf \\
        [--recipe q4_k|q8_0] [--layers 4] [--no-mtp] [--keep-inert] \\
        [--dry-run] [--jobs N]

Why `qwen35`. Agnes's text decoder is Qwen3.5's dense decoder (its
`modeling_agnes.py`: Gated DeltaNet layers with a gated global-attention
layer every fourth, one-centred RMSNorms, a sigmoid output gate produced
with the queries, partial interleaved MRoPE, one MTP layer) with one
addition: every layer's MLP carries a second, narrower SwiGLU
(`mlp.parallel_ffn`, 2048 wide) on the same input, summed into the
residual.

What the Preview checkpoint computes. Not all of it is trained (measured
2026-11-17, `find_inert`): every layer's parallel branch, and the whole of
layers 20-23 and 48-51 (two periods of three GDN layers and an attention
layer), hold initial values (gate and up projections of RMS 0.02 with no
zeros, where trained ones are 30 % zeros) with output projections near
1e-30. Those add nothing to the residual stream, so the converter leaves
them out by default: the file is a 64-layer Qwen3.5 dense model with a
17408-wide MLP (Qwen3.5-27B's shape), 3.0 GB smaller at `q4_k` and as much
less read a decoded token, computing the same function to within ~1e-12
of a residual of order 1. Nothing is left out on a guess: a block only
when every output projection it adds to the residual is below INERT_MAX,
the branch only when its down projection is in every layer; anything else
is converted. `--keep-inert` converts everything.

A live parallel branch is folded into the main MLP. Two SwiGLUs on one
input are one SwiGLU of the summed width: the gate and up rows
concatenated, the down columns concatenated, since `down_a(silu(g_a) *
u_a) + down_b(silu(g_b) * u_b) = [down_a | down_b] (silu([g_a; g_b]) *
[u_a; u_b])`, which is what Agnes's own SGLang patch does
(`sglang_patch/README.md`). Either way the result is an ordinary Qwen3.5
dense model, which llama.cpp runs as it is (a reference for lily) and
lily's Qwen3.5-family kernels read.

The conversion follows llama.cpp's own for Qwen3.5 (`conversion/qwen.py`,
`Qwen3_5TextModel` and its bases), which defines what the file's tensors
mean:

- RMSNorm weights are stored as `1 + w` (Agnes's `AgnesRMSNorm` scales by
  `1 + w`), except the GDN's gated norm (`delta_attn.norm`, a plain `w`).
  The addition is done in f32 after widening the bf16 source.
- `A_log` is stored as `ssm_a = -exp(A_log)`; `dt_bias` as `ssm_dt.bias`;
  the conv kernel `[ch, 1, 4]` as `[ch, 4]` f32.
- The GDN's 48 value heads are reordered from Hugging Face's grouped order
  (value head `h` reads key head `h / 3`) to the tiled order ggml
  broadcasts in (key head `h % 16`; lily's `qwen36.comp` reads it so):
  the value rows of `in_proj_qkv` and the conv, `in_proj_z`, `in_proj_a`,
  `in_proj_b`, `A_log`, `dt_bias`, and the columns of `out_proj`.
- Separate `attn_qkv`/`attn_gate` and `ssm_alpha`/`ssm_beta` tensors
  (llama.cpp's loader takes them as well as its merged forms; lily's
  kernels read them separately).
- The MTP block is `blk.<n>` (after the n trunk blocks written) with the
  `nextn.*` tensors, in its own
  file (llama.cpp's `--mtp-only` output: it also carries the embedding, the
  output norm and the LM head, which lily does not map).

The MTP layer's MLP has no parallel branch (17408 wide). With a folded
trunk (19456) it is padded with zero rows (gate, up) and zero columns
(down) so every block has one width. The padding is exact: a padded unit
computes `silu(0) * 0 = 0`, and its quantized blocks are zero blocks that
decode to exact zeros.

Recipes (the vision tower is not converted; lily serves text):

- `q4_k` (default, 15.9 GB trunk, the 36 GB machines' setting): Q4_K for
  every matrix except `attn_k`, `attn_v` and `output`, which are Q8_0
  (llama.cpp's Q4_K_M keeps those at higher precision too; lily's
  projection kernels read Q4_K and Q8_0). Tensors read in one fused
  dispatch share a type (`schema.rs`'s fused groups).
- `q8_0` (28.6 GB trunk): every matrix Q8_0.

Vectors, norms and the conv kernel stay f32. The quantizers are ports of
ggml's reference `quantize_row_q8_0_ref` and `quantize_row_q4_K_ref`
(`ggml/src/ggml-quants.c`, MIT), in f32 with ggml's summation order:
`llama-quantize` without an importance matrix calls the same functions.
Every tensor's quantization error (relative RMS) is measured as it is
written and the worst ones are reported at the end.

Only numpy is needed. `--layers N` converts the first N source layers
(less any inert ones: the 4-layer checkpoint the tests and the Hugging Face
comparison use); the MTP block follows the blocks written.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

# gguf_type
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = range(8)
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(8, 13)

# ggml_type: (id, values a block, bytes a block)
GGML_TYPES = {
    "F32": (0, 1, 4),
    "Q8_0": (8, 32, 34),
    "Q4_K": (12, 256, 144),
}

# llama_ftype, for display only (llama.cpp prints it; nothing reads it).
FILE_TYPES = {"q8_0": 7, "q4_k": 15}

TOKEN_NORMAL, TOKEN_CONTROL, TOKEN_USER_DEFINED, TOKEN_UNUSED = 1, 3, 4, 5


class Meta:
    """GGUF key/values in insertion order, each with its explicit type."""

    def __init__(self) -> None:
        self.items: list[tuple[str, int, object, int | None]] = []

    def add(self, key: str, ty: int, value: object, sub: int | None = None) -> None:
        if any(k == key for k, *_ in self.items):
            raise ValueError(f"duplicate GGUF key {key}")
        self.items.append((key, ty, value, sub))

    def u32(self, key: str, v: int) -> None:
        self.add(key, T_UINT32, int(v))

    def f32(self, key: str, v: float) -> None:
        self.add(key, T_FLOAT32, float(v))

    def boolean(self, key: str, v: bool) -> None:
        self.add(key, T_BOOL, bool(v))

    def string(self, key: str, v: str) -> None:
        self.add(key, T_STRING, v)

    def array(self, key: str, sub: int, v: list) -> None:
        self.add(key, T_ARRAY, list(v), sub)


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


_SCALAR = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}


def _gguf_value(ty: int, v: object, sub: int | None) -> bytes:
    if ty == T_STRING:
        return _gguf_string(v)  # type: ignore[arg-type]
    if ty == T_ARRAY:
        assert sub is not None
        items: list = v  # type: ignore[assignment]
        head = struct.pack("<IQ", sub, len(items))
        if sub == T_STRING:
            return head + b"".join(_gguf_string(s) for s in items)
        return head + np.asarray(items, dtype=np.dtype(_SCALAR[sub])).tobytes()
    return struct.pack(_SCALAR[ty], v)


@dataclass
class OutTensor:
    """One tensor of an output file: its GGUF name, ggml dims (innermost
    first), type, and how to produce its f32 values (`rows x cols`, or a
    vector)."""

    name: str
    dims: tuple[int, ...]
    ty: str
    produce: Callable[[], np.ndarray]
    source: str

    @property
    def nbytes(self) -> int:
        _, per, size = GGML_TYPES[self.ty]
        n = math.prod(self.dims)
        if n % per:
            raise ValueError(f"{self.name}: {n} values are not whole {self.ty} blocks")
        if self.dims[0] % per:
            raise ValueError(f"{self.name}: rows of {self.dims[0]} values for {self.ty}")
        return n // per * size


def _pad(n: int) -> int:
    return (ALIGNMENT - n % ALIGNMENT) % ALIGNMENT


def gguf_header(meta: Meta, tensors: list[OutTensor]) -> tuple[bytes, list[int]]:
    """The header up to the aligned start of the data, and each tensor's
    offset relative to that start."""
    out = bytearray(GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(meta.items)))
    for key, ty, v, sub in meta.items:
        out += _gguf_string(key) + struct.pack("<I", ty) + _gguf_value(ty, v, sub)
    offsets = []
    at = 0
    for t in tensors:
        offsets.append(at)
        out += _gguf_string(t.name) + struct.pack("<I", len(t.dims))
        out += struct.pack(f"<{len(t.dims)}Q", *t.dims)
        out += struct.pack("<IQ", GGML_TYPES[t.ty][0], at)
        at += t.nbytes + _pad(t.nbytes)
    out += b"\0" * _pad(len(out))
    return bytes(out), offsets


def read_gguf(path: Path) -> tuple[dict[str, object], dict[str, tuple[tuple[int, ...], int, int]], int]:
    """A GGUF's metadata, its tensor table (name -> dims, ggml type id,
    offset) and the data's start: the converter's own check of what it
    wrote."""
    with open(path, "rb") as f:
        # Mapped, not read: only the header's pages are touched.
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    at = 0

    def take(fmt: str):
        nonlocal at
        v = struct.unpack_from(fmt, data, at)
        at += struct.calcsize(fmt)
        return v if len(v) > 1 else v[0]

    def string() -> str:
        nonlocal at
        n = take("<Q")
        s = data[at : at + n].decode("utf-8")
        at += n
        return s

    def value(ty: int):
        if ty == T_STRING:
            return string()
        if ty == T_ARRAY:
            sub, n = take("<IQ")
            return [value(sub) for _ in range(n)]
        return take(_SCALAR[ty])

    if data[:4] != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF")
    at = 4
    version, n_tensors, n_kv = take("<IQQ")
    if version != GGUF_VERSION:
        raise ValueError(f"{path}: GGUF version {version}")
    meta: dict[str, object] = {}
    for _ in range(n_kv):
        key = string()
        meta[key] = value(take("<I"))
    table = {}
    for _ in range(n_tensors):
        name = string()
        nd = take("<I")
        dims = tuple(struct.unpack_from(f"<{nd}Q", data, at))
        at += 8 * nd
        ty, off = take("<IQ")
        table[name] = (dims, ty, off)
    start = at + _pad(at)
    return meta, table, start


# ---------------------------------------------------------------------------
# Quantization: ports of ggml's reference quantizers
# ---------------------------------------------------------------------------


def _seqsum(a: np.ndarray) -> np.ndarray:
    """Sum over axis 0 in index order (ggml's loops), not numpy's pairwise
    order: the quantizers' choices depend on the last bits."""
    s = a[0].copy()
    for i in range(1, a.shape[0]):
        s += a[i]
    return s


def _round_away(v: np.ndarray) -> np.ndarray:
    """C's `roundf` (half away from zero), exact: in f64, where `|v| + 0.5`
    of an f32 is exact."""
    v = v.astype(np.float64)
    return np.copysign(np.floor(np.abs(v) + 0.5), v)


def quantize_q8_0(x: np.ndarray) -> bytes:
    """`quantize_row_q8_0_ref` over `x`'s rows (f32, a multiple of 32
    values each)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = _round_away(x * inv[:, None]).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(b: bytes, n: int) -> np.ndarray:
    blocks = np.frombuffer(b, dtype=np.uint8).reshape(-1, 34)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)[:n]


def _make_qkx2_quants(x: np.ndarray, w: np.ndarray, nmax: int, rmin: float, rdelta: float, nstep: int):
    """`make_qkx2_quants` (use_mad false) over columns: `x`, `w` are
    `[32][n]` (one sub-block a column). Returns (scale, the_min, L)."""
    f = np.float32
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    sum_w = _seqsum(w)
    sum_x = _seqsum(w * x)
    mn = np.where(mn > 0, f(0), mn)
    flat = mx == mn
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        iscale = f(nmax) / (mx - mn)
        scale = f(1) / iscale
        L = np.clip(np.rint(iscale * (x - mn)), 0, nmax)
        diff = scale * L + mn - x
        best = _seqsum(w * (diff * diff))
        for step in range(nstep + 1):
            iscale = (f(rmin) + f(rdelta) * f(step) + f(nmax)) / (mx - mn)
            laux = np.clip(np.rint(iscale * (x - mn)), 0, nmax)
            wl = w * laux
            sum_l = _seqsum(wl)
            sum_l2 = _seqsum(wl * laux)
            sum_xl = _seqsum(wl * x)
            det = sum_w * sum_l2 - sum_l * sum_l
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / det
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det
            pos = this_min > 0
            this_min = np.where(pos, f(0), this_min)
            this_scale = np.where(pos, sum_xl / sum_l2, this_scale)
            diff = this_scale * laux + this_min - x
            cur = _seqsum(w * (diff * diff))
            better = (det > 0) & (cur < best) & ~flat
            L = np.where(better, laux, L)
            best = np.where(better, cur, best)
            scale = np.where(better, this_scale, scale)
            mn = np.where(better, this_min, mn)
    scale = np.where(flat, f(0), scale).astype(np.float32)
    L = np.where(flat, f(0), L)
    return scale, (-mn).astype(np.float32), L


def quantize_q4_k(x: np.ndarray) -> bytes:
    """`quantize_row_q4_K_ref` over `x`'s rows (f32, a multiple of 256
    values each)."""
    f = np.float32
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 256)
    nb = x.shape[0]
    xs = x.reshape(nb * 8, 32).T  # [32][sub-blocks]
    av_x = np.sqrt(_seqsum(xs * xs) / f(32))
    w = av_x + np.abs(xs)
    scales, mins, L = _make_qkx2_quants(xs, w, 15, -1.0, 0.1, 20)
    scales = scales.reshape(nb, 8)
    mins = mins.reshape(nb, 8)
    max_scale = np.maximum(scales.max(axis=1), f(0))
    max_min = np.maximum(mins.max(axis=1), f(0))
    with np.errstate(divide="ignore"):
        inv_scale = np.where(max_scale > 0, f(63) / max_scale, f(0)).astype(np.float32)
        inv_min = np.where(max_min > 0, f(63) / max_min, f(0)).astype(np.float32)
    ls = np.minimum(np.rint(inv_scale[:, None] * scales), 63).astype(np.uint8)
    lm = np.minimum(np.rint(inv_min[:, None] * mins), 63).astype(np.uint8)
    sc = np.zeros((nb, 12), dtype=np.uint8)
    sc[:, 0:4] = ls[:, 0:4]
    sc[:, 4:8] = lm[:, 0:4]
    sc[:, 8:12] = (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4)
    sc[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
    sc[:, 4:8] |= (lm[:, 4:8] >> 4) << 6
    d16 = (max_scale / f(63)).astype(np.float16)
    dmin16 = (max_min / f(63)).astype(np.float16)
    # The final codes against the rounded scales (get_scale_min_k4 returns
    # ls and lm as packed); a zero scale keeps make_qkx2_quants's codes.
    dj = d16.astype(np.float32)[:, None] * ls.astype(np.float32)
    dmj = dmin16.astype(np.float32)[:, None] * lm.astype(np.float32)
    xb = x.reshape(nb, 8, 32)
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.clip(np.rint((xb + dmj[:, :, None]) / dj[:, :, None]), 0, 15)
    L = L.T.reshape(nb, 8, 32)
    L = np.where((dj != 0)[:, :, None], l2, L).astype(np.uint8).reshape(nb, 256)
    # qs: each 64 values as 32 bytes, low nibbles the first 32.
    Lq = L.reshape(nb, 4, 2, 32)
    qs = (Lq[:, :, 0, :] | (Lq[:, :, 1, :] << 4)).reshape(nb, 128)
    out = np.empty((nb, 144), dtype=np.uint8)
    out[:, 0:2] = d16.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = dmin16.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = sc
    out[:, 16:] = qs
    return out.tobytes()


def dequantize_q4_k(b: bytes, n: int) -> np.ndarray:
    """`dequantize_row_q4_K`."""
    blocks = np.frombuffer(b, dtype=np.uint8).reshape(-1, 144)
    nb = blocks.shape[0]
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32)
    dmin = blocks[:, 2:4].copy().view("<f2").astype(np.float32)
    s = blocks[:, 4:16]
    sc = np.empty((nb, 8), dtype=np.uint8)
    m = np.empty((nb, 8), dtype=np.uint8)
    sc[:, 0:4] = s[:, 0:4] & 63
    m[:, 0:4] = s[:, 4:8] & 63
    sc[:, 4:8] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    q = blocks[:, 16:].reshape(nb, 4, 32)
    L = np.stack([q & 0xF, q >> 4], axis=2).reshape(nb, 8, 32).astype(np.float32)
    y = (d * sc.astype(np.float32))[:, :, None] * L - (dmin * m.astype(np.float32))[:, :, None]
    return y.reshape(-1)[:n]


QUANTIZERS = {"Q8_0": (quantize_q8_0, dequantize_q8_0), "Q4_K": (quantize_q4_k, dequantize_q4_k)}


def encode(x: np.ndarray, ty: str) -> tuple[bytes, float, float]:
    """`x` (f32) as `ty`, with the squared error of the round trip and the
    squared norm (for the relative RMS error)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if ty == "F32":
        return x.tobytes(), 0.0, float(np.sum(np.square(x, dtype=np.float64)))
    q, dq = QUANTIZERS[ty]
    b = q(x)
    err = (dq(b, x.size) - x.ravel()).astype(np.float64)
    # Plain numpy sums, not np.dot: BLAS would start threads in every worker.
    return b, float(np.sum(err * err)), float(np.sum(np.square(x, dtype=np.float64)))


def _encode_job(args: tuple[np.ndarray, str]) -> tuple[bytes, float, float]:
    return encode(*args)


# ---------------------------------------------------------------------------
# The source checkpoint
# ---------------------------------------------------------------------------

_ST_DTYPES = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}


@dataclass
class SourceTensor:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int

    def load(self) -> np.ndarray:
        """The tensor as f32 (bf16 widened exactly)."""
        n = math.prod(self.shape)
        a = np.memmap(self.path, dtype=_ST_DTYPES[self.dtype], mode="r", offset=self.offset, shape=(n,))
        if self.dtype == "BF16":
            # Widened in place: one f32-sized buffer, not two (the
            # embedding is 5.1 GB as f32).
            w = a.astype(np.uint32)
            w <<= 16
            return w.view(np.float32).reshape(self.shape)
        return np.asarray(a.astype(np.float32)).reshape(self.shape)

    def absmax(self) -> float:
        """The largest magnitude, streamed in pieces (a NaN reads as
        infinite: never inert)."""
        n = math.prod(self.shape)
        a = np.memmap(self.path, dtype=_ST_DTYPES[self.dtype], mode="r", offset=self.offset, shape=(n,))
        top = 0.0
        for i in range(0, n, 1 << 24):
            piece = a[i : i + (1 << 24)]
            if self.dtype == "BF16":
                # Magnitude bits order like the magnitudes; NaN sorts above inf.
                bits = int((piece & 0x7FFF).max()) if piece.size else 0
                v = float(np.array([bits << 16], np.uint32).view(np.float32)[0])
                top = max(top, math.inf if math.isnan(v) else v)
            else:
                v = float(np.abs(piece.astype(np.float32)).max()) if piece.size else 0.0
                top = max(top, math.inf if math.isnan(v) else v)
        return top


def read_source(src: Path) -> dict[str, SourceTensor]:
    """Every tensor of the checkpoint's shards (from the index, or the
    single `model.safetensors`), by name."""
    index = src / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    out: dict[str, SourceTensor] = {}
    for fname in files:
        path = src / fname
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if info["dtype"] not in _ST_DTYPES:
                raise ValueError(f"{name}: dtype {info['dtype']} is not supported")
            if name in out:
                raise ValueError(f"tensor {name} appears in two shards")
            begin, end = info["data_offsets"]
            t = SourceTensor(name, path, info["dtype"], tuple(info["shape"]), 8 + n + begin)
            if end - begin != math.prod(t.shape) * np.dtype(_ST_DTYPES[t.dtype]).itemsize:
                raise ValueError(f"{name}: {end - begin} bytes for shape {t.shape}")
            out[name] = t
    return out


# ---------------------------------------------------------------------------
# The model's shape and the transforms
# ---------------------------------------------------------------------------

DELTA, GLOBAL = "agnes_delta_attention", "agnes_global_attention"


@dataclass
class Shape:
    n_layer: int
    hidden: int
    vocab: int
    ff: int
    ff_par: int
    n_head: int
    n_head_kv: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    rope_sections: list[int]
    rms_eps: float
    k_heads: int
    v_heads: int
    gdn_dim: int
    conv: int
    recurrent: list[bool]
    context: int
    mtp: int
    # The source layers converted, in order (blk.i is source layer keep[i]),
    # and whether the parallel branch is folded in (else dropped as inert).
    keep: list[int] = field(default_factory=list)
    fold: bool = True

    @property
    def ff_total(self) -> int:
        return self.ff + (self.ff_par if self.fold else 0)

    @property
    def n_out(self) -> int:
        """Trunk blocks in the output."""
        return len(self.keep)

    @property
    def recurrent_out(self) -> list[bool]:
        return [self.recurrent[il] for il in self.keep]


def read_shape(cfg: dict, layers: int | None) -> Shape:
    """The text config, checked against what this conversion implements:
    anything else is refused rather than converted into a model that
    computes something different."""
    if cfg.get("model_type") != "agnes":
        raise ValueError(f"model_type {cfg.get('model_type')!r}, expected 'agnes'")
    t = cfg["text_config"]
    rope = t.get("rope_parameters", {})
    expect = {
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "tie_word_embeddings": False,
        "mtp_use_dedicated_embeddings": False,
    }
    for k, v in expect.items():
        if t.get(k, v) != v:
            raise ValueError(f"text_config.{k} is {t.get(k)!r}; this converter implements {v!r}")
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"rope_type {rope.get('rope_type')!r} is not supported")
    if not rope.get("mrope_interleaved", True):
        raise ValueError("non-interleaved MRoPE is not qwen35's")
    if t["linear_key_head_dim"] != t["linear_value_head_dim"]:
        raise ValueError("GDN key and value head widths differ")
    if t["linear_num_value_heads"] % t["linear_num_key_heads"]:
        raise ValueError("GDN value heads are not a multiple of the key heads")
    if t.get("mtp_num_hidden_layers", 0) not in (0, 1):
        raise ValueError(f"{t['mtp_num_hidden_layers']} MTP layers (qwen35 has at most one)")
    types = t["layer_types"]
    n_all = t["num_hidden_layers"]
    if len(types) != n_all or set(types) - {DELTA, GLOBAL}:
        raise ValueError(f"layer_types {sorted(set(types))} over {len(types)} of {n_all} layers")
    n = n_all if layers is None else layers
    if not 1 <= n <= n_all:
        raise ValueError(f"--layers {n} of {n_all}")
    partial = rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
    head_dim = t["head_dim"]
    sections = list(rope.get("mrope_section", [11, 11, 10]))
    return Shape(
        n_layer=n,
        hidden=t["hidden_size"],
        vocab=t["vocab_size"],
        ff=t["intermediate_size"],
        ff_par=t.get("parallel_ffn_intermediate_size", 0) or 0,
        n_head=t["num_attention_heads"],
        n_head_kv=t["num_key_value_heads"],
        head_dim=head_dim,
        rope_dim=int(head_dim * partial),
        rope_theta=float(rope.get("rope_theta", t.get("rope_theta"))),
        rope_sections=(sections + [0, 0, 0, 0])[:4],
        rms_eps=float(t["rms_norm_eps"]),
        k_heads=t["linear_num_key_heads"],
        v_heads=t["linear_num_value_heads"],
        gdn_dim=t["linear_key_head_dim"],
        conv=t["linear_conv_kernel_dim"],
        recurrent=[ty == DELTA for ty in types[:n]],
        context=t["max_position_embeddings"],
        mtp=t.get("mtp_num_hidden_layers", 0),
        keep=list(range(n)),
        fold=bool(t.get("parallel_ffn_intermediate_size", 0)),
    )


# A tensor whose every value is below this contributes nothing: at most
# 1e-20 * |activation| * width, some 1e-12 for this model's activations and
# widths, against a residual stream of order 1; and f16 block scales cannot
# hold it (the smallest is 6e-8), so any GGUF quantization stores it as zeros.
INERT_MAX = 1e-20


def find_inert(src: dict[str, SourceTensor], s: Shape) -> tuple[list[int], bool, list[str]]:
    """The blocks and the parallel branch that compute nothing: (source
    layers kept, whether to fold the parallel branch, the source tensors
    left out). A block is left out only when every output projection it
    adds to the residual (the mixer's, the MLP's down, the parallel
    branch's down) is inert; the parallel branch only when its down is
    inert in every layer. The Preview checkpoint (2026-11-17): layers
    20-23 and 48-51 and every layer's parallel branch hold initial values
    with down/output projections near 1e-30, so it computes as a 64-layer
    model with a 17408-wide MLP."""
    p = lambda il: f"model.language_model.layers.{il}."  # noqa: E731

    def inert(name: str) -> bool:
        return src[name].absmax() < INERT_MAX if name in src else False

    par_down = [p(il) + "mlp.parallel_ffn.down_proj.weight" for il in range(s.n_layer)]
    fold = bool(s.ff_par) and not all(inert(n) for n in par_down)
    keep, dropped = [], []
    for il in range(s.n_layer):
        mixer = p(il) + ("delta_attn.out_proj.weight" if s.recurrent[il] else "global_attn.o_proj.weight")
        outs = [mixer, p(il) + "mlp.down_proj.weight"] + ([par_down[il]] if fold else [])
        if all(inert(n) for n in outs):
            dropped += [n for n in src if n.startswith(p(il))]
        else:
            keep.append(il)
    if not keep:
        raise ValueError("every block is inert: nothing to convert")
    if s.ff_par and not fold:
        dropped += [n for n in src if ".mlp.parallel_ffn." in n and n.startswith("model.language_model.layers.")]
    return keep, fold, sorted(set(dropped))


def reorder_v_heads(a: np.ndarray, axis: int, k_heads: int, v_per_k: int, head_dim: int) -> np.ndarray:
    """Value heads along `axis` from grouped order (`[k0: v0..v{r-1}, k1:
    ...]`) to tiled (`[v0 of k0, v0 of k1, ..., v1 of k0, ...]`):
    llama.cpp's `_reorder_v_heads`."""
    axis %= a.ndim
    shape = a.shape
    g = a.reshape(shape[:axis] + (k_heads, v_per_k, head_dim) + shape[axis + 1 :])
    g = np.swapaxes(g, axis, axis + 1)
    return np.ascontiguousarray(g).reshape(shape)


# ---------------------------------------------------------------------------
# The plan: every output tensor, its source and type
# ---------------------------------------------------------------------------


def tensor_type(name: str, rows_of: int, recipe: str) -> str:
    """The storage type of a 2D tensor by its GGUF name (vectors, norms and
    the conv kernel are f32 regardless)."""
    if recipe == "q8_0":
        return "Q8_0"
    if name.endswith(("attn_k.weight", "attn_v.weight")) or name == "output.weight":
        return "Q8_0"
    if rows_of % 256:
        raise ValueError(f"{name}: rows of {rows_of} values are not whole Q4_K blocks")
    return "Q4_K"


class Planner:
    """Builds the output tensors of the trunk and MTP files from the source
    tensors, recording which source tensors it used."""

    def __init__(self, src: dict[str, SourceTensor], shape: Shape, recipe: str) -> None:
        self.src = src
        self.s = shape
        self.recipe = recipe
        self.used: set[str] = set()

    def get(self, name: str, shape: tuple[int, ...]) -> SourceTensor:
        t = self.src.get(name)
        if t is None:
            raise ValueError(f"the checkpoint has no tensor {name}")
        if t.shape != shape:
            raise ValueError(f"{name} has shape {t.shape}, expected {shape}")
        self.used.add(name)
        return t

    def matrix(self, gname: str, fn: Callable[[], np.ndarray], rows: int, cols: int, source: str) -> OutTensor:
        return OutTensor(gname, (cols, rows), tensor_type(gname, cols, self.recipe), fn, source)

    def vector(self, gname: str, fn: Callable[[], np.ndarray], n: int, source: str) -> OutTensor:
        return OutTensor(gname, (n,), "F32", fn, source)

    def plain(self, gname: str, sname: str, rows: int, cols: int) -> OutTensor:
        t = self.get(sname, (rows, cols))
        return self.matrix(gname, t.load, rows, cols, sname)

    def norm(self, gname: str, sname: str, n: int, one_centred: bool = True) -> OutTensor:
        t = self.get(sname, (n,))
        fn = (lambda: t.load() + np.float32(1)) if one_centred else t.load
        return self.vector(gname, fn, n, sname)

    def attention(self, il: int, p: str) -> list[OutTensor]:
        s = self.s
        b = f"blk.{il}."
        q = s.n_head * s.head_dim
        kv = s.n_head_kv * s.head_dim
        return [
            self.plain(b + "attn_q.weight", p + "q_proj.weight", 2 * q, s.hidden),
            self.plain(b + "attn_k.weight", p + "k_proj.weight", kv, s.hidden),
            self.plain(b + "attn_v.weight", p + "v_proj.weight", kv, s.hidden),
            self.plain(b + "attn_output.weight", p + "o_proj.weight", s.hidden, q),
            self.norm(b + "attn_q_norm.weight", p + "q_norm.weight", s.head_dim),
            self.norm(b + "attn_k_norm.weight", p + "k_norm.weight", s.head_dim),
        ]

    def delta(self, il: int, p: str) -> list[OutTensor]:
        s = self.s
        b = f"blk.{il}."
        r = s.v_heads // s.k_heads
        kd = s.k_heads * s.gdn_dim
        vd = s.v_heads * s.gdn_dim
        ch = 2 * kd + vd
        rv = lambda a, axis, hd: reorder_v_heads(a, axis, s.k_heads, r, hd)  # noqa: E731
        qkv = self.get(p + "in_proj_qkv.weight", (ch, s.hidden))
        z = self.get(p + "in_proj_z.weight", (vd, s.hidden))
        a = self.get(p + "in_proj_a.weight", (s.v_heads, s.hidden))
        bt = self.get(p + "in_proj_b.weight", (s.v_heads, s.hidden))
        a_log = self.get(p + "A_log", (s.v_heads,))
        dt = self.get(p + "dt_bias", (s.v_heads,))
        conv = self.get(p + "conv1d.weight", (ch, 1, s.conv))
        out = self.get(p + "out_proj.weight", (s.hidden, vd))

        def qkv_fn() -> np.ndarray:
            w = qkv.load()
            return np.concatenate([w[: 2 * kd], rv(w[2 * kd :], 0, s.gdn_dim)])

        def conv_fn() -> np.ndarray:
            w = conv.load().reshape(ch, s.conv)
            return np.concatenate([w[: 2 * kd], rv(w[2 * kd :], 0, s.gdn_dim)])

        def ssm_a_fn() -> np.ndarray:
            return rv(-np.exp(a_log.load())[:, None], 0, 1)[:, 0]

        return [
            self.matrix(b + "attn_qkv.weight", qkv_fn, ch, s.hidden, qkv.name),
            self.matrix(b + "attn_gate.weight", lambda: rv(z.load(), 0, s.gdn_dim), vd, s.hidden, z.name),
            self.matrix(b + "ssm_alpha.weight", lambda: rv(a.load(), 0, 1), s.v_heads, s.hidden, a.name),
            self.matrix(b + "ssm_beta.weight", lambda: rv(bt.load(), 0, 1), s.v_heads, s.hidden, bt.name),
            self.vector(b + "ssm_a", ssm_a_fn, s.v_heads, a_log.name),
            self.vector(b + "ssm_dt.bias", lambda: rv(dt.load()[:, None], 0, 1)[:, 0], s.v_heads, dt.name),
            OutTensor(b + "ssm_conv1d.weight", (s.conv, ch), "F32", conv_fn, conv.name),
            self.norm(b + "ssm_norm.weight", p + "norm.weight", s.gdn_dim, one_centred=False),
            self.matrix(b + "ssm_out.weight", lambda: rv(out.load(), 1, s.gdn_dim), s.hidden, vd, out.name),
        ]

    def mlp(self, il: int, p: str, parallel: bool) -> list[OutTensor]:
        """The MLP at the trunk's width: with the branch folded, its rows
        and columns appended (trunk) or zeros in their place (MTP); with the
        branch dropped, the main MLP alone."""
        s = self.s
        b = f"blk.{il}."
        h, ff = s.hidden, s.ff
        fp = s.ff_par if s.fold else 0
        main = {k: self.get(f"{p}{k}_proj.weight", (h, ff) if k == "down" else (ff, h)) for k in ("gate", "up", "down")}
        par = (
            {k: self.get(f"{p}parallel_ffn.{k}_proj.weight", (h, fp) if k == "down" else (fp, h)) for k in ("gate", "up", "down")}
            if parallel and fp
            else None
        )
        if not fp:
            return [
                self.plain(b + "ffn_gate.weight", main["gate"].name, ff, h),
                self.plain(b + "ffn_up.weight", main["up"].name, ff, h),
                self.plain(b + "ffn_down.weight", main["down"].name, h, ff),
            ]

        def rows(k: str) -> Callable[[], np.ndarray]:
            def fn() -> np.ndarray:
                extra = par[k].load() if par else np.zeros((fp, h), np.float32)
                return np.concatenate([main[k].load(), extra], axis=0)

            return fn

        def down() -> np.ndarray:
            extra = par["down"].load() if par else np.zeros((h, fp), np.float32)
            return np.concatenate([main["down"].load(), extra], axis=1)

        src = lambda k: main[k].name + (" + parallel_ffn" if par else " + zeros")  # noqa: E731
        return [
            self.matrix(b + "ffn_gate.weight", rows("gate"), s.ff_total, h, src("gate")),
            self.matrix(b + "ffn_up.weight", rows("up"), s.ff_total, h, src("up")),
            self.matrix(b + "ffn_down.weight", down, h, s.ff_total, src("down")),
        ]

    def block(self, il: int, p: str, recurrent: bool, parallel: bool) -> list[OutTensor]:
        s = self.s
        b = f"blk.{il}."
        v = [
            self.norm(b + "attn_norm.weight", p + "input_layernorm.weight", s.hidden),
            self.norm(b + "post_attention_norm.weight", p + "post_attention_layernorm.weight", s.hidden),
        ]
        v += self.delta(il, p + "delta_attn.") if recurrent else self.attention(il, p + "global_attn.")
        return v + self.mlp(il, p + "mlp.", parallel)

    def head(self) -> list[OutTensor]:
        s = self.s
        return [
            self.plain("token_embd.weight", "model.language_model.embed_tokens.weight", s.vocab, s.hidden),
            self.norm("output_norm.weight", "model.language_model.norm.weight", s.hidden),
            self.plain("output.weight", "lm_head.weight", s.vocab, s.hidden),
        ]

    def trunk(self) -> list[OutTensor]:
        v = self.head()
        for out, il in enumerate(self.s.keep):
            v += self.block(out, f"model.language_model.layers.{il}.", self.s.recurrent[il], True)
        return v

    def mtp(self) -> list[OutTensor]:
        s = self.s
        il = s.n_out
        b = f"blk.{il}.nextn."
        v = self.head()
        v += self.block(il, "mtp.layers.0.", False, False)
        v += [
            self.plain(b + "eh_proj.weight", "mtp.fc.weight", s.hidden, 2 * s.hidden),
            self.norm(b + "enorm.weight", "mtp.pre_fc_norm_embedding.weight", s.hidden),
            self.norm(b + "hnorm.weight", "mtp.pre_fc_norm_hidden.weight", s.hidden),
            self.norm(b + "shared_head_norm.weight", "mtp.norm.weight", s.hidden),
        ]
        return v


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def tokenizer_meta(src: Path, vocab: int, meta: Meta) -> None:
    """The tokenizer as llama.cpp's converter writes a Qwen3.5 one
    (`get_vocab_base`, `SpecialVocab`): byte-level BPE with the `qwen35`
    pre-tokenizer, added tokens as control (special, or `<|...|>`) or
    user-defined, `[PAD<id>]` up to the embedding's rows."""
    tj = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    model = tj["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"tokenizer model {model.get('type')}, expected BPE")
    if tj.get("normalizer") != {"type": "NFC"}:
        raise ValueError(f"normalizer {tj.get('normalizer')}: qwen35's is NFC")
    if tj.get("pre_tokenizer") != QWEN35_PRE_TOKENIZER:
        raise ValueError("the pre-tokenizer is not Qwen3.5's split and byte-level step")
    tokens: dict[int, str] = {i: t for t, i in model["vocab"].items()}
    types: dict[int, int] = {i: TOKEN_NORMAL for i in tokens}
    for a in tj["added_tokens"]:
        i, text = a["id"], a["content"]
        if i in tokens and tokens[i] != text:
            raise ValueError(f"added token {i} {text!r} collides with {tokens[i]!r}")
        tokens[i] = text
        special = a["special"] or (text.startswith("<|") and text.endswith("|>"))
        types[i] = TOKEN_CONTROL if special else TOKEN_USER_DEFINED
    if max(tokens) >= vocab:
        raise ValueError(f"token id {max(tokens)} outside the vocabulary of {vocab}")
    ids = {t: i for i, t in tokens.items()}
    # Pairs (transformers >= 4.45) with a space inside a part encode it as
    # U+0120, as gguf-py's SpecialVocab does, so the line stays unambiguous.
    merges = [
        m if isinstance(m, str) else " ".join(part.replace(" ", "\u0120") for part in m)
        for m in model["merges"]
    ]
    meta.string("tokenizer.ggml.model", "gpt2")
    meta.string("tokenizer.ggml.pre", "qwen35")
    meta.array("tokenizer.ggml.tokens", T_STRING, [tokens.get(i, f"[PAD{i}]") for i in range(vocab)])
    meta.array("tokenizer.ggml.token_type", T_INT32, [types.get(i, TOKEN_UNUSED) for i in range(vocab)])
    meta.array("tokenizer.ggml.merges", T_STRING, merges)
    # SpecialVocab: tokenizer_config's tokens first, then config.json's ids.
    def special(key: str) -> int:
        entry = tc.get(key)
        text = entry.get("content") if isinstance(entry, dict) else entry
        if not isinstance(text, str) or text not in ids:
            raise ValueError(f"tokenizer_config.json {key} {entry!r} is not a token")
        return ids[text]

    eos = special("eos_token")
    pad = special("pad_token")
    cfg = json.loads((src / "config.json").read_text())
    bos = cfg.get("bos_token_id", cfg["text_config"].get("bos_token_id"))
    if not isinstance(bos, int) or not 0 <= bos < vocab:
        raise ValueError(f"bos_token_id {bos!r}")
    meta.u32("tokenizer.ggml.bos_token_id", bos)
    meta.u32("tokenizer.ggml.eos_token_id", eos)
    meta.u32("tokenizer.ggml.padding_token_id", pad)
    meta.boolean("tokenizer.ggml.add_bos_token", bool(tc.get("add_bos_token", False)))
    meta.boolean("tokenizer.ggml.add_eos_token", bool(tc.get("add_eos_token", False)))
    tmpl = src / "chat_template.jinja"
    template = tmpl.read_text(encoding="utf-8") if tmpl.is_file() else tc.get("chat_template")
    if not isinstance(template, str):
        raise ValueError("no chat template (chat_template.jinja or tokenizer_config.json)")
    meta.string("tokenizer.chat_template", template)
    # The NFC normalizer (checked above) and the decoder's space clean-up, as
    # ggml-org's Qwen3.6 GGUFs carry them.
    meta.boolean("tokenizer.ggml.normalizer.nfc", True)
    meta.boolean("tokenizer.ggml.clean_spaces", bool(tc.get("clean_up_tokenization_spaces", False)))


# The qwen35 pre-tokenizer's split (`src/qwen36/tokenizer.rs`).
QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Qwen3.5's `tokenizer.json` pre-tokenizer, which `qwen35` implements.
QWEN35_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": QWEN35_SPLIT}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    ],
}

ARCH = "qwen35"


def model_meta(s: Shape, src: Path, recipe: str, name: str, mtp_file: bool) -> Meta:
    m = Meta()
    a = lambda k: f"{ARCH}.{k}"  # noqa: E731
    m.string("general.architecture", ARCH)
    m.string("general.name", name)
    m.u32("general.file_type", FILE_TYPES[recipe])
    m.u32("general.quantization_version", 2)
    n = s.n_out + (1 if mtp_file else 0)
    m.u32(a("block_count"), n)
    m.u32(a("context_length"), s.context)
    m.u32(a("embedding_length"), s.hidden)
    m.u32(a("feed_forward_length"), s.ff_total)
    m.u32(a("attention.head_count"), s.n_head)
    m.u32(a("attention.head_count_kv"), s.n_head_kv)
    m.f32(a("rope.freq_base"), s.rope_theta)
    m.f32(a("attention.layer_norm_rms_epsilon"), s.rms_eps)
    m.u32(a("attention.key_length"), s.head_dim)
    m.u32(a("attention.value_length"), s.head_dim)
    m.u32(a("rope.dimension_count"), s.rope_dim)
    m.array(a("rope.dimension_sections"), T_INT32, s.rope_sections)
    m.u32(a("ssm.conv_kernel"), s.conv)
    m.u32(a("ssm.state_size"), s.gdn_dim)
    m.u32(a("ssm.group_count"), s.k_heads)
    m.u32(a("ssm.time_step_rank"), s.v_heads)
    m.u32(a("ssm.inner_size"), s.v_heads * s.gdn_dim)
    m.array(a("attention.recurrent_layers"), T_BOOL, s.recurrent_out + [False] * (n - s.n_out))
    # The interval only where the written trunk follows it: a reader that
    # derives the layer kinds from it alone (when the list is absent, or
    # ignored) must not get another pattern than the list's. Leaving out
    # blocks that are not whole periods breaks it; the Preview's (20-23,
    # 48-51) are whole periods.
    if s.recurrent_out == [(i + 1) % 4 != 0 for i in range(s.n_out)]:
        m.u32(a("full_attention_interval"), 4)
    if mtp_file:
        m.u32(a("nextn_predict_layers"), 1)
    tokenizer_meta(src, s.vocab, m)
    return m


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


# Values a worker quantizes at a time. Q4_K's search makes some 300 passes
# over a dozen temporaries of its input's size, so the chunk must stay in
# cache: on x86-64 (32 cores, 2026-11-17) 2^16 values a chunk ran 94 M values
# a second over 14 workers and 110 M over 32, 2^20 32 M, 2^22 8 M (DRAM
# bound, no faster with more workers). It also bounds a worker's memory.
CHUNK_VALUES = 1 << 16


def _chunks(x: np.ndarray, ty: str) -> list[np.ndarray]:
    """Views of `x`'s rows of at most CHUNK_VALUES values (vectors and
    f32 tensors whole)."""
    if ty == "F32" or x.ndim == 1:
        return [x]
    per = max(1, CHUNK_VALUES // x.shape[1])
    return [x[i : i + per] for i in range(0, x.shape[0], per)]


def _encoded(parts: list[np.ndarray], ty: str, pool: ProcessPoolExecutor | None, window: int):
    """`encode` of each part, in order, with at most `window` parts in
    flight (each submitted part is a pickled copy of its rows)."""
    if pool is None or len(parts) == 1:
        yield from (encode(p, ty) for p in parts)
        return
    pending: deque = deque()
    for p in parts:
        if len(pending) >= window:
            yield pending.popleft().result()
        pending.append(pool.submit(_encode_job, (p, ty)))
    while pending:
        yield pending.popleft().result()


def write_gguf(
    path: Path,
    meta: Meta,
    tensors: list[OutTensor],
    pool: ProcessPoolExecutor | None,
    jobs: int,
    errors: list[tuple[float, str, str]],
) -> None:
    header, offsets = gguf_header(meta, tensors)
    tmp = path.with_suffix(path.suffix + ".part")
    t0 = time.time()
    with open(tmp, "wb") as f:
        f.write(header)
        base = f.tell()
        for i, (t, off) in enumerate(zip(tensors, offsets)):
            x = t.produce()
            want = tuple(reversed(t.dims)) if len(t.dims) > 1 else t.dims
            if x.shape != want:
                raise ValueError(f"{t.name}: produced {x.shape}, declared {want}")
            if not np.all(np.isfinite(x)):
                raise ValueError(f"{t.name}: non-finite values from {t.source}")
            err = norm = 0.0
            if f.tell() != base + off:
                raise ValueError(f"{t.name} at {f.tell() - base}, planned {off}")
            for b, e, n2 in _encoded(_chunks(x, t.ty), t.ty, pool, 4 * jobs):
                f.write(b)
                err += e
                norm += n2
            f.write(b"\0" * _pad(t.nbytes))
            if f.tell() - base != off + t.nbytes + _pad(t.nbytes):
                raise ValueError(f"{t.name}: wrote {f.tell() - base - off} bytes, planned {t.nbytes}")
            if t.ty != "F32":
                errors.append((math.sqrt(err / norm) if norm else 0.0, t.name, t.ty))
            print(
                f"  [{i + 1}/{len(tensors)}] {t.name} {t.ty} {list(t.dims)} "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
    verify(tmp, meta, tensors)
    os.replace(tmp, path)


def verify(path: Path, meta: Meta, tensors: list[OutTensor]) -> None:
    """Reads the written file back: metadata, tensor table, file size."""
    got_meta, table, start = read_gguf(path)
    if list(got_meta) != [k for k, *_ in meta.items]:
        raise ValueError(f"{path}: metadata keys differ from what was written")
    if list(table) != [t.name for t in tensors]:
        raise ValueError(f"{path}: tensor table differs from what was written")
    end = 0
    for t in tensors:
        dims, ty, off = table[t.name]
        if dims != t.dims or ty != GGML_TYPES[t.ty][0] or off % ALIGNMENT:
            raise ValueError(f"{path}: {t.name} is {dims} type {ty} at {off}")
        end = max(end, off + t.nbytes + _pad(t.nbytes))
    if path.stat().st_size != start + end:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {start + end}")


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the Hugging Face checkpoint directory")
    ap.add_argument("--dst", type=Path, required=True, help="output directory (created)")
    ap.add_argument("--recipe", choices=sorted(FILE_TYPES), default="q4_k")
    ap.add_argument("--layers", type=int, help="convert only the first N trunk layers")
    ap.add_argument("--no-mtp", action="store_true", help="do not write the MTP file")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and sizes, write nothing")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="worker processes")
    ap.add_argument(
        "--keep-inert",
        action="store_true",
        help="convert inert blocks and fold an inert parallel branch instead of leaving them out",
    )
    args = ap.parse_args(argv)

    cfg = json.loads((args.src / "config.json").read_text())
    shape = read_shape(cfg, args.layers)
    src = read_source(args.src)
    dropped: list[str] = []
    if not args.keep_inert:
        shape.keep, shape.fold, dropped = find_inert(src, shape)
        gone = sorted(set(range(shape.n_layer)) - set(shape.keep))
        if gone:
            print(f"inert blocks left out (every output projection below {INERT_MAX:g}): source layers {gone}")
        if shape.ff_par and not shape.fold:
            print(f"inert parallel MLP branch left out in every layer: MLP {shape.ff_total} wide")
        if shape.ff_par and shape.fold:
            print(f"parallel MLP branch folded in: MLP {shape.ff_total} wide")
    planner = Planner(src, shape, args.recipe)
    files: list[tuple[str, Meta, list[OutTensor]]] = []
    tag = args.recipe.upper() + (f"-L{shape.n_layer}" if args.layers else "")
    base = f"Agnes-3.0-Flash-{tag}"
    name = "Agnes 3.0 Flash" + (f" (first {shape.n_layer} layers)" if args.layers else "")
    files.append((f"{base}.gguf", model_meta(shape, args.src, args.recipe, name, False), planner.trunk()))
    if not args.no_mtp:
        if not shape.mtp:
            raise ValueError("the checkpoint has no MTP layer; pass --no-mtp")
        files.append((f"mtp-{base}.gguf", model_meta(shape, args.src, args.recipe, name + " MTP", True), planner.mtp()))

    # Every text tensor of the checkpoint must be consumed: a revision with
    # tensors this conversion does not know is refused, not half-converted.
    skipped = {n for n in src if n.startswith("model.visual.")}
    if args.layers:
        skipped |= {
            n for n in src
            if n.startswith("model.language_model.layers.") and int(n.split(".")[3]) >= shape.n_layer
        }
    if args.no_mtp:
        skipped |= {n for n in src if n.startswith("mtp.")}
    unused = sorted(set(src) - planner.used - skipped - set(dropped))
    if unused:
        raise ValueError(f"{len(unused)} source tensors are not converted, e.g. {unused[:5]}")

    for fname, _, tensors in files:
        by_type: dict[str, int] = {}
        for t in tensors:
            by_type[t.ty] = by_type.get(t.ty, 0) + t.nbytes
        detail = ", ".join(f"{k} {fmt_gb(v)}" for k, v in sorted(by_type.items()))
        print(f"{fname}: {len(tensors)} tensors, {fmt_gb(sum(by_type.values()))} ({detail})")
    if args.dry_run:
        return

    # lily (`Qwen36Model::find_mtp`) takes the first `mtp-*.gguf` beside a
    # model: another conversion's MTP file in the same directory could be
    # taken instead of this one's. One directory a conversion.
    ours = {fname for fname, _, _ in files}
    others = sorted(p.name for p in args.dst.glob("mtp-*.gguf") if p.name not in ours) if args.dst.is_dir() else []
    if others:
        raise ValueError(f"{args.dst} already holds {others}; use a directory of its own for each conversion")
    # Every output is checked before any is written: a trunk written beside
    # an old MTP file of the same name would be served with it.
    existing = [str(args.dst / fname) for fname, _, _ in files if (args.dst / fname).exists()]
    if existing:
        raise ValueError(f"{', '.join(existing)} exist; remove them first")
    args.dst.mkdir(parents=True, exist_ok=True)
    errors: list[tuple[float, str, str]] = []
    jobs = max(1, args.jobs)
    with ProcessPoolExecutor(jobs) if jobs > 1 else _NoPool() as pool:
        for fname, meta, tensors in files:
            out = args.dst / fname
            print(f"writing {out}")
            write_gguf(out, meta, tensors, pool if jobs > 1 else None, jobs, errors)
    errors.sort(reverse=True)
    print("largest relative RMS quantization errors:")
    for e, n, ty in errors[:8]:
        print(f"  {e:.4f} {n} {ty}")


class _NoPool:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"error: {e}")
