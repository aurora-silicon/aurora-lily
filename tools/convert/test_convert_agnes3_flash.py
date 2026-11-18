"""Unit tests for the Agnes 3.0 Flash converter; numpy only, no checkpoint.

    .venv/bin/python tools/convert/test_convert_agnes3_flash.py

- The quantizers against a literal scalar port of ggml's reference C (the
  vectorized code must make the same choices bit for bit), and their round
  trips.
- A synthetic Agnes checkpoint (4 layers, small widths) converted end to
  end: the file's metadata, tensor table and types, and the transforms
  checked by running the model's pieces both ways. A Gated DeltaNet layer
  over a few tokens in Hugging Face's convention (grouped value heads,
  `1 + w` norms) on the source weights must equal the same layer in
  llama.cpp's (tiled heads, `h % k_heads`, stored norms) on the converted
  weights; so must the MLP with its parallel branch against the folded one.
  Those run on an f32 conversion (exact); the recipes' types and the
  quantized round trip are checked separately.
"""

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import convert_agnes3_flash as conv  # noqa: E402

f32 = np.float32


# ---------------------------------------------------------------------------
# The quantizers
# ---------------------------------------------------------------------------


def nearest_int(v: f32) -> int:
    """ggml's `nearest_int`: the f32 sum with 1.5 * 2^23 rounds to even."""
    s = np.float32(v) + np.float32(12582912.0)
    return int(s.view(np.int32) & 0x007FFFFF) - 0x00400000


def ref_make_qkx2(x, w, nmax, rmin, rdelta, nstep):
    """`make_qkx2_quants` transliterated, one f32 operation at a time."""
    n = len(x)
    mn, mx = x[0], x[0]
    sum_w = w[0]
    sum_x = sum_w * x[0]
    for i in range(1, n):
        mn = min(mn, x[i]) if x[i] < mn else mn
        mx = x[i] if x[i] > mx else mx
        sum_w = f32(sum_w + w[i])
        sum_x = f32(sum_x + f32(w[i] * x[i]))
    if mn > 0:
        mn = f32(0)
    L = [0] * n
    if mx == mn:
        return f32(0), f32(-mn), L
    iscale = f32(f32(nmax) / f32(mx - mn))
    scale = f32(f32(1) / iscale)
    best = f32(0)
    for i in range(n):
        L[i] = max(0, min(nmax, nearest_int(f32(iscale * f32(x[i] - mn)))))
        d = f32(f32(f32(scale * f32(L[i])) + mn) - x[i])
        best = f32(best + f32(w[i] * f32(d * d)))
    for step in range(nstep + 1):
        iscale = f32(f32(f32(f32(rmin) + f32(f32(rdelta) * f32(step))) + f32(nmax)) / f32(mx - mn))
        laux = []
        sl = sl2 = sxl = f32(0)
        for i in range(n):
            q = max(0, min(nmax, nearest_int(f32(iscale * f32(x[i] - mn)))))
            laux.append(q)
            wl = f32(w[i] * f32(q))
            sl = f32(sl + wl)
            sl2 = f32(sl2 + f32(wl * f32(q)))
            sxl = f32(sxl + f32(wl * x[i]))
        det = f32(f32(sum_w * sl2) - f32(sl * sl))
        if det > 0:
            ts = f32(f32(f32(sum_w * sxl) - f32(sum_x * sl)) / det)
            tm = f32(f32(f32(sl2 * sum_x) - f32(sl * sxl)) / det)
            if tm > 0:
                tm = f32(0)
                ts = f32(sxl / sl2)
            cur = f32(0)
            for i in range(n):
                d = f32(f32(f32(ts * f32(laux[i])) + tm) - x[i])
                cur = f32(cur + f32(w[i] * f32(d * d)))
            if cur < best:
                L, best, scale, mn = laux, cur, ts, tm
    return scale, f32(-mn), L


def ref_q4_k_block(x: np.ndarray) -> bytes:
    """`quantize_row_q4_K_ref` for one block of 256 values."""
    scales, mins, Ls = [], [], []
    for j in range(8):
        xs = x[32 * j : 32 * j + 32]
        s2 = f32(0)
        for v in xs:
            s2 = f32(s2 + f32(v * v))
        av = np.sqrt(f32(s2 / f32(32)), dtype=np.float32)
        w = [f32(av + abs(v)) for v in xs]
        s, m, L = ref_make_qkx2(list(xs), w, 15, -1.0, 0.1, 20)
        scales.append(s)
        mins.append(m)
        Ls.append(L)
    max_scale = max([f32(0)] + [s for s in scales if s > 0])
    max_min = max([f32(0)] + [m for m in mins if m > 0])
    inv_s = f32(f32(63) / max_scale) if max_scale > 0 else f32(0)
    inv_m = f32(f32(63) / max_min) if max_min > 0 else f32(0)
    sc = [0] * 12
    ls_all, lm_all = [], []
    for j in range(8):
        ls = min(63, nearest_int(f32(inv_s * scales[j])) & 0xFF)
        lm = min(63, nearest_int(f32(inv_m * mins[j])) & 0xFF)
        ls_all.append(ls)
        lm_all.append(lm)
        if j < 4:
            sc[j] = ls
            sc[j + 4] = lm
        else:
            sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
            sc[j - 4] |= (ls >> 4) << 6
            sc[j] |= (lm >> 4) << 6
    d16 = np.float16(f32(max_scale / f32(63)))
    dmin16 = np.float16(f32(max_min / f32(63)))
    L = [v for Lj in Ls for v in Lj]
    for j in range(8):
        d = f32(f32(d16) * f32(ls_all[j]))
        if not d:
            continue
        dm = f32(f32(dmin16) * f32(lm_all[j]))
        for i in range(32):
            L[32 * j + i] = max(0, min(15, nearest_int(f32(f32(x[32 * j + i] + dm) / d))))
    qs = []
    for j in range(0, 256, 64):
        qs += [L[j + i] | (L[j + i + 32] << 4) for i in range(32)]
    return d16.tobytes() + dmin16.tobytes() + bytes(sc) + bytes(qs)


def ref_q8_0_block(x: np.ndarray) -> bytes:
    amax = f32(0)
    for v in x:
        amax = max(amax, f32(abs(v)))
    d = f32(amax / f32(127))
    inv = f32(f32(1) / d) if d else f32(0)
    q = [int(math.copysign(math.floor(abs(float(f32(v * inv))) + 0.5), float(f32(v * inv)))) for v in x]
    return np.float16(d).tobytes() + np.array(q, dtype=np.int8).tobytes()


class Quantizers(unittest.TestCase):
    def blocks(self) -> list[np.ndarray]:
        rng = np.random.default_rng(3)
        out = [
            rng.standard_normal(256).astype(f32) * f32(0.02),
            (rng.standard_t(2, 256) * 0.05).astype(f32),
            np.abs(rng.standard_normal(256)).astype(f32) + f32(0.5),
            -np.abs(rng.standard_normal(256)).astype(f32),
            np.zeros(256, f32),
            np.full(256, 0.25, f32),
            (np.round(rng.standard_normal(256) * 4) / 4).astype(f32),
        ]
        mixed = rng.standard_normal(256).astype(f32)
        mixed[:32] = 0
        mixed[100] = 30
        out.append(mixed)
        return out

    def test_q4_k_matches_the_scalar_reference(self):
        for i, x in enumerate(self.blocks()):
            self.assertEqual(conv.quantize_q4_k(x), ref_q4_k_block(x), f"block {i}")

    def test_q8_0_matches_the_scalar_reference(self):
        for i, x in enumerate(self.blocks()):
            want = b"".join(ref_q8_0_block(x[k : k + 32]) for k in range(0, 256, 32))
            self.assertEqual(conv.quantize_q8_0(x), want, f"block {i}")

    def test_round_trips(self):
        x = np.random.default_rng(5).standard_normal((64, 512)).astype(f32)
        for ty, bound in (("Q8_0", 0.01), ("Q4_K", 0.12)):
            b, err, norm = conv.encode(x, ty)
            self.assertEqual(len(b), x.size // conv.GGML_TYPES[ty][1] * conv.GGML_TYPES[ty][2])
            rel = math.sqrt(err / norm)
            self.assertLess(rel, bound, ty)
            back = conv.QUANTIZERS[ty][1](b, x.size).reshape(x.shape)
            self.assertAlmostEqual(float(np.sum((back - x) ** 2)), err, delta=err * 1e-3)

    def test_zero_blocks_decode_to_exact_zeros(self):
        z = np.zeros((4, 512), f32)
        for ty in ("Q8_0", "Q4_K"):
            b, _, _ = conv.encode(z, ty)
            self.assertTrue(np.all(conv.QUANTIZERS[ty][1](b, z.size) == 0), ty)


# ---------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------


class Reorder(unittest.TestCase):
    def test_tiled_order_pairs_value_heads_with_key_head_mod(self):
        k_heads, r, hd = 4, 3, 2
        # Row value: the HF value head's key head (grouped: h // r).
        hf = np.repeat(np.arange(k_heads * r) // r, hd).astype(f32)
        tiled = conv.reorder_v_heads(hf[:, None], 0, k_heads, r, hd)[:, 0]
        for p in range(k_heads * r):
            self.assertTrue(np.all(tiled[p * hd : (p + 1) * hd] == p % k_heads))

    def test_columns_and_rows_agree(self):
        a = np.arange(5 * 12, dtype=f32).reshape(5, 12)
        rows = conv.reorder_v_heads(a.T.copy(), 0, 2, 3, 2)
        cols = conv.reorder_v_heads(a, 1, 2, 3, 2)
        np.testing.assert_array_equal(rows.T, cols)


# ---------------------------------------------------------------------------
# A synthetic checkpoint
# ---------------------------------------------------------------------------

H, FF, FP, NH, NKV, HD = 256, 256, 256, 2, 1, 128
KH, VH, GD, CONV = 2, 6, 128, 4  # 3 value heads a key head, as Agnes
VOCAB = 320
LAYERS = 4


def text_config() -> dict:
    return {
        "model_type": "agnes_text",
        "hidden_size": H,
        "num_hidden_layers": LAYERS,
        "intermediate_size": FF,
        "parallel_ffn_intermediate_size": FP,
        "vocab_size": VOCAB,
        "layer_types": [conv.DELTA] * 3 + [conv.GLOBAL],
        "num_attention_heads": NH,
        "num_key_value_heads": NKV,
        "head_dim": HD,
        "attention_bias": False,
        "attn_output_gate": True,
        "linear_num_key_heads": KH,
        "linear_num_value_heads": VH,
        "linear_key_head_dim": GD,
        "linear_value_head_dim": GD,
        "linear_conv_kernel_dim": CONV,
        "output_gate_type": "swish",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "max_position_embeddings": 4096,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "bos_token_id": 300,
        "eos_token_id": 300,
        "tie_word_embeddings": False,
    }


def source_tensors(rng) -> dict[str, np.ndarray]:
    t: dict[str, np.ndarray] = {}
    w = lambda *s: (rng.standard_normal(s) * 0.05).astype(f32)  # noqa: E731
    kd, vd = KH * GD, VH * GD

    def attn(p):
        t[p + "q_proj.weight"] = w(2 * NH * HD, H)
        t[p + "k_proj.weight"] = w(NKV * HD, H)
        t[p + "v_proj.weight"] = w(NKV * HD, H)
        t[p + "o_proj.weight"] = w(H, NH * HD)
        t[p + "q_norm.weight"] = w(HD)
        t[p + "k_norm.weight"] = w(HD)

    def mlp(p, ff, par):
        t[p + "gate_proj.weight"] = w(ff, H)
        t[p + "up_proj.weight"] = w(ff, H)
        t[p + "down_proj.weight"] = w(H, ff)
        if par:
            t[p + "parallel_ffn.gate_proj.weight"] = w(FP, H)
            t[p + "parallel_ffn.up_proj.weight"] = w(FP, H)
            t[p + "parallel_ffn.down_proj.weight"] = w(H, FP)

    for il in range(LAYERS):
        p = f"model.language_model.layers.{il}."
        t[p + "input_layernorm.weight"] = w(H)
        t[p + "post_attention_layernorm.weight"] = w(H)
        if il % 4 != 3:
            d = p + "delta_attn."
            t[d + "in_proj_qkv.weight"] = w(2 * kd + vd, H)
            t[d + "in_proj_z.weight"] = w(vd, H)
            t[d + "in_proj_a.weight"] = w(VH, H)
            t[d + "in_proj_b.weight"] = w(VH, H)
            t[d + "A_log"] = (rng.random(VH) * 2).astype(f32)
            t[d + "dt_bias"] = rng.standard_normal(VH).astype(f32)
            t[d + "conv1d.weight"] = w(2 * kd + vd, 1, CONV)
            t[d + "norm.weight"] = (1 + w(GD)).astype(f32)
            t[d + "out_proj.weight"] = w(H, vd)
        else:
            attn(p + "global_attn.")
        mlp(p + "mlp.", FF, True)
    t["model.language_model.embed_tokens.weight"] = w(VOCAB, H)
    t["model.language_model.norm.weight"] = w(H)
    t["lm_head.weight"] = w(VOCAB, H)
    m = "mtp.layers.0."
    t[m + "input_layernorm.weight"] = w(H)
    t[m + "post_attention_layernorm.weight"] = w(H)
    attn(m + "global_attn.")
    mlp(m + "mlp.", FF, False)
    t["mtp.fc.weight"] = w(H, 2 * H)
    t["mtp.pre_fc_norm_embedding.weight"] = w(H)
    t["mtp.pre_fc_norm_hidden.weight"] = w(H)
    t["mtp.norm.weight"] = w(H)
    t["model.visual.blocks.0.attn.qkv.weight"] = w(8, 8)
    return t


def bf16(a: np.ndarray) -> np.ndarray:
    """Values made exactly representable in bf16 (what the checkpoint holds)."""
    return (a.astype(f32).view(np.uint32) & 0xFFFF0000).view(f32)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, blobs, at = {}, [], 0
    for name, a in tensors.items():
        raw = (a.astype(f32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def write_checkpoint(d: Path, tensors: dict[str, np.ndarray]) -> None:
    (d / "config.json").write_text(json.dumps({"model_type": "agnes", "text_config": text_config()}))
    write_safetensors(d / "model.safetensors", tensors)
    vocab = {f"t{i}": i for i in range(300)}
    added = [
        {"id": 300, "content": "<|endoftext|>", "special": True},
        {"id": 301, "content": "<|im_start|>", "special": True},
        {"id": 302, "content": "<|im_end|>", "special": True},
        {"id": 303, "content": "<think>", "special": False},
    ]
    (d / "tokenizer.json").write_text(json.dumps({
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": conv.QWEN35_PRE_TOKENIZER,
        "added_tokens": added,
        "model": {"type": "BPE", "vocab": vocab, "merges": ["t1 t2", ["t3", "t4"], ["t 5", "t6"]]},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "<|im_end|>", "pad_token": {"content": "<|endoftext|>"}, "add_bos_token": False,
        "clean_up_tokenization_spaces": False,
    }))
    (d / "chat_template.jinja").write_text("{{ messages }}")


def load_gguf_tensor(path: Path, name: str) -> np.ndarray:
    """A tensor of a converted file, decoded to f32 in HF shape."""
    _, table, start = conv.read_gguf(path)
    dims, ty, off = table[name]
    kind = next(k for k, v in conv.GGML_TYPES.items() if v[0] == ty)
    n = math.prod(dims)
    _, per, size = conv.GGML_TYPES[kind]
    with open(path, "rb") as f:
        f.seek(start + off)
        raw = f.read(n // per * size)
    x = np.frombuffer(raw, f32).copy() if kind == "F32" else conv.QUANTIZERS[kind][1](raw, n)
    return x.reshape(tuple(reversed(dims)))


# The f32 reference forward pieces.

def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(x))


def l2n(x):
    return x / np.sqrt(np.sum(x * x, -1, keepdims=True) + 1e-6)


def delta_layer(x, w, pairing):
    """One GDN layer over the rows of `x` (tokens), stepwise: q/k/v from
    the conv, the gated delta rule, the gated norm, the output projection.
    `pairing(h)` is the key head value head `h` reads; `w` holds HF-shaped
    weights with the norm as the multiplier it applies."""
    T = x.shape[0]
    kd = KH * GD
    qkv = x @ w["qkv"].T
    z = (x @ w["z"].T).reshape(T, VH, GD)
    a = x @ w["a"].T
    b = x @ w["b"].T
    padded = np.concatenate([np.zeros((CONV - 1, qkv.shape[1])), qkv])
    c = silu(np.stack([np.sum(padded[t : t + CONV].T * w["conv"], -1) for t in range(T)]))
    q = l2n(c[:, :kd].reshape(T, KH, GD)) / math.sqrt(GD)
    k = l2n(c[:, kd : 2 * kd].reshape(T, KH, GD))
    v = c[:, 2 * kd :].reshape(T, VH, GD)
    g = w["ssm_a"] * softplus(a + w["dt"])  # ssm_a = -exp(A_log)
    beta = 1 / (1 + np.exp(-b))
    S = np.zeros((VH, GD, GD))
    out = np.zeros((T, VH, GD))
    for t in range(T):
        for h in range(VH):
            kh = pairing(h)
            S[h] *= math.exp(g[t, h])
            rec = k[t, kh] @ S[h]
            S[h] += np.outer(k[t, kh], (v[t, h] - rec) * beta[t, h])
            out[t, h] = q[t, kh] @ S[h]
    y = rmsnorm(out, w["norm"]) * silu(z)
    return y.reshape(T, -1) @ w["out"].T


class Conversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.src = root / "src"
        cls.src.mkdir()
        cls.tensors = {k: bf16(v) for k, v in source_tensors(np.random.default_rng(11)).items()}
        write_checkpoint(cls.src, cls.tensors)
        cls.out = {}
        for recipe in ("q4_k", "q8_0"):
            dst = root / recipe
            conv.main(["--src", str(cls.src), "--dst", str(dst), "--recipe", recipe, "--jobs", "2"])
            cls.out[recipe] = dst
        # An f32 conversion for the exact checks.
        saved = conv.tensor_type
        conv.tensor_type = lambda name, cols, recipe: "F32"
        try:
            conv.main(["--src", str(cls.src), "--dst", str(root / "f32"), "--jobs", "1"])
        finally:
            conv.tensor_type = saved
        cls.f32 = root / "f32" / "Agnes-3.0-Flash-Q4_K.gguf"
        cls.f32_mtp = root / "f32" / "mtp-Agnes-3.0-Flash-Q4_K.gguf"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metadata(self):
        meta, table, _ = conv.read_gguf(self.out["q4_k"] / "Agnes-3.0-Flash-Q4_K.gguf")
        a = lambda k: meta["qwen35." + k]  # noqa: E731
        self.assertEqual(meta["general.architecture"], "qwen35")
        self.assertEqual(a("block_count"), LAYERS)
        self.assertEqual(a("feed_forward_length"), FF + FP)
        self.assertEqual(a("attention.recurrent_layers"), [True, True, True, False])
        self.assertEqual(a("full_attention_interval"), 4)
        self.assertEqual(a("rope.dimension_count"), HD // 4)
        self.assertEqual(a("rope.dimension_sections"), [11, 11, 10, 0])
        self.assertEqual((a("ssm.group_count"), a("ssm.time_step_rank"), a("ssm.inner_size")), (KH, VH, VH * GD))
        self.assertEqual((a("ssm.state_size"), a("ssm.conv_kernel")), (GD, CONV))
        self.assertNotIn("qwen35.nextn_predict_layers", meta)
        tokens = meta["tokenizer.ggml.tokens"]
        types = meta["tokenizer.ggml.token_type"]
        self.assertEqual(len(tokens), VOCAB)
        self.assertEqual((tokens[302], types[302]), ("<|im_end|>", conv.TOKEN_CONTROL))
        self.assertEqual((tokens[303], types[303]), ("<think>", conv.TOKEN_USER_DEFINED))
        self.assertEqual((tokens[310], types[310]), ("[PAD310]", conv.TOKEN_UNUSED))
        # A space inside a pair's part is encoded as U+0120 (gguf-py's SpecialVocab).
        self.assertEqual(meta["tokenizer.ggml.merges"], ["t1 t2", "t3 t4", "t\u01205 t6"])
        self.assertEqual(meta["tokenizer.ggml.add_eos_token"], False)
        self.assertEqual(meta["tokenizer.ggml.normalizer.nfc"], True)
        self.assertEqual(meta["tokenizer.ggml.clean_spaces"], False)
        self.assertEqual(meta["tokenizer.ggml.eos_token_id"], 302)
        self.assertEqual(meta["tokenizer.ggml.padding_token_id"], 300)
        self.assertEqual(meta["tokenizer.ggml.pre"], "qwen35")
        self.assertNotIn("blk.0.ffn_gate_par.weight", table)
        mtp_meta, mtp_table, _ = conv.read_gguf(self.out["q4_k"] / "mtp-Agnes-3.0-Flash-Q4_K.gguf")
        self.assertEqual(mtp_meta["qwen35.block_count"], LAYERS + 1)
        self.assertEqual(mtp_meta["qwen35.nextn_predict_layers"], 1)
        self.assertEqual(mtp_meta["qwen35.attention.recurrent_layers"], [True, True, True, False, False])
        self.assertIn(f"blk.{LAYERS}.nextn.eh_proj.weight", mtp_table)
        self.assertFalse(any(n.startswith("blk.0.") for n in mtp_table))

    def test_recipe_types(self):
        ids = {v[0]: k for k, v in conv.GGML_TYPES.items()}
        _, t4, _ = conv.read_gguf(self.out["q4_k"] / "Agnes-3.0-Flash-Q4_K.gguf")
        _, t8, _ = conv.read_gguf(self.out["q8_0"] / "Agnes-3.0-Flash-Q8_0.gguf")
        ty4 = {n: ids[v[1]] for n, v in t4.items()}
        ty8 = {n: ids[v[1]] for n, v in t8.items()}
        self.assertEqual(ty4["blk.3.attn_k.weight"], "Q8_0")
        self.assertEqual(ty4["blk.3.attn_v.weight"], "Q8_0")
        self.assertEqual(ty4["output.weight"], "Q8_0")
        self.assertEqual(ty4["blk.0.ffn_down.weight"], "Q4_K")
        self.assertEqual(ty4["token_embd.weight"], "Q4_K")
        for n in ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"):
            self.assertEqual(ty4[f"blk.0.{n}.weight"], "Q4_K", n)
        for n in ("blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_norm.weight", "output_norm.weight"):
            self.assertEqual((ty4[n], ty8[n]), ("F32", "F32"), n)
        self.assertEqual({t for n, t in ty8.items() if not t == "F32"}, {"Q8_0"})

    def test_quantized_values_track_the_source(self):
        got = load_gguf_tensor(self.out["q8_0"] / "Agnes-3.0-Flash-Q8_0.gguf", "blk.3.attn_q.weight")
        want = self.tensors["model.language_model.layers.3.global_attn.q_proj.weight"]
        self.assertLess(np.abs(got - want).max(), 0.01)

    def test_norms_are_stored_one_centred_except_the_gated_norm(self):
        p = "model.language_model.layers."
        got = load_gguf_tensor(self.f32, "blk.0.attn_norm.weight")
        np.testing.assert_array_equal(got, self.tensors[p + "0.input_layernorm.weight"] + f32(1))
        got = load_gguf_tensor(self.f32, "blk.3.attn_k_norm.weight")
        np.testing.assert_array_equal(got, self.tensors[p + "3.global_attn.k_norm.weight"] + f32(1))
        got = load_gguf_tensor(self.f32, "blk.0.ssm_norm.weight")
        np.testing.assert_array_equal(got, self.tensors[p + "0.delta_attn.norm.weight"])
        got = load_gguf_tensor(self.f32_mtp, f"blk.{LAYERS}.nextn.enorm.weight")
        np.testing.assert_array_equal(got, self.tensors["mtp.pre_fc_norm_embedding.weight"] + f32(1))

    def test_delta_layer_computes_the_same_both_ways(self):
        p = "model.language_model.layers.1.delta_attn."
        s = lambda n: self.tensors[p + n].astype(np.float64)  # noqa: E731
        hf = {
            "qkv": s("in_proj_qkv.weight"), "z": s("in_proj_z.weight"),
            "a": s("in_proj_a.weight"), "b": s("in_proj_b.weight"),
            # -exp(A_log) in f32, as the converter (and llama.cpp's) stores it.
            "conv": s("conv1d.weight")[:, 0, :],
            "ssm_a": (-np.exp(self.tensors[p + "A_log"])).astype(np.float64),
            "dt": s("dt_bias"), "norm": s("norm.weight"), "out": s("out_proj.weight"),
        }
        g = lambda n: load_gguf_tensor(self.f32, f"blk.1.{n}").astype(np.float64)  # noqa: E731
        gg = {
            "qkv": g("attn_qkv.weight"), "z": g("attn_gate.weight"),
            "a": g("ssm_alpha.weight"), "b": g("ssm_beta.weight"),
            "conv": g("ssm_conv1d.weight"), "ssm_a": g("ssm_a"),
            "dt": g("ssm_dt.bias"), "norm": g("ssm_norm.weight"), "out": g("ssm_out.weight"),
        }
        x = np.random.default_rng(2).standard_normal((5, H))
        want = delta_layer(x, hf, lambda h: h // (VH // KH))
        got = delta_layer(x, gg, lambda h: h % KH)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
        # The pairing matters: the converted weights in HF's pairing differ.
        self.assertGreater(np.abs(delta_layer(x, gg, lambda h: h // (VH // KH)) - want).max(), 1e-4)

    def test_mlp_fold_computes_the_same(self):
        p = "model.language_model.layers.2.mlp."
        s = lambda n: self.tensors[p + n].astype(np.float64)  # noqa: E731
        x = np.random.default_rng(4).standard_normal((3, H))
        want = (silu(x @ s("gate_proj.weight").T) * (x @ s("up_proj.weight").T)) @ s("down_proj.weight").T
        want += (silu(x @ s("parallel_ffn.gate_proj.weight").T) * (x @ s("parallel_ffn.up_proj.weight").T)) @ s(
            "parallel_ffn.down_proj.weight"
        ).T
        g = lambda n: load_gguf_tensor(self.f32, f"blk.2.{n}").astype(np.float64)  # noqa: E731
        got = (silu(x @ g("ffn_gate.weight").T) * (x @ g("ffn_up.weight").T)) @ g("ffn_down.weight").T
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)

    def test_mtp_mlp_is_padded_with_exact_zeros(self):
        for recipe, base in (("q4_k", "Q4_K"), ("q8_0", "Q8_0")):
            path = self.out[recipe] / f"mtp-Agnes-3.0-Flash-{base}.gguf"
            gate = load_gguf_tensor(path, f"blk.{LAYERS}.ffn_gate.weight")
            down = load_gguf_tensor(path, f"blk.{LAYERS}.ffn_down.weight")
            self.assertEqual(gate.shape, (FF + FP, H))
            self.assertTrue(np.all(gate[FF:] == 0) and np.all(down[:, FF:] == 0), recipe)
            self.assertTrue(np.any(gate[:FF] != 0))

    def test_layers_option_and_unknown_tensors(self):
        root = Path(self.tmp.name)
        conv.main(["--src", str(self.src), "--dst", str(root / "l2"), "--layers", "2", "--no-mtp", "--jobs", "1"])
        meta, table, _ = conv.read_gguf(root / "l2" / "Agnes-3.0-Flash-Q4_K-L2.gguf")
        self.assertEqual(meta["qwen35.block_count"], 2)
        self.assertFalse(any(n.startswith("blk.2.") for n in table))
        self.assertFalse((root / "l2" / "mtp-Agnes-3.0-Flash-Q4_K-L2.gguf").exists())
        extra = root / "extra"
        extra.mkdir()
        t = dict(self.tensors)
        t["model.language_model.layers.0.mlp.surprise.weight"] = np.zeros((2, 2), f32)
        write_checkpoint(extra, t)
        with self.assertRaisesRegex(ValueError, "not converted"):
            conv.main(["--src", str(extra), "--dst", str(root / "x"), "--dry-run"])

    def test_writes_nothing_when_an_output_exists(self):
        # An old MTP file of the same name: refused before the trunk is written.
        dst = Path(self.tmp.name) / "stale"
        dst.mkdir()
        (dst / "mtp-Agnes-3.0-Flash-Q4_K.gguf").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "exist"):
            conv.main(["--src", str(self.src), "--dst", str(dst), "--jobs", "1"])
        self.assertFalse((dst / "Agnes-3.0-Flash-Q4_K.gguf").exists())

    def test_refuses_a_directory_with_another_mtp_file(self):
        # lily takes the first mtp-*.gguf beside a model.
        with self.assertRaisesRegex(ValueError, "directory of its own"):
            conv.main(["--src", str(self.src), "--dst", str(self.out["q4_k"]), "--recipe", "q8_0", "--jobs", "1"])

    def test_refuses_what_it_does_not_implement(self):
        cfg = {"model_type": "agnes", "text_config": dict(text_config(), output_gate_type="sigmoid")}
        with self.assertRaisesRegex(ValueError, "output_gate_type"):
            conv.read_shape(cfg, None)
        cfg = {"model_type": "agnes", "text_config": dict(text_config(), tie_word_embeddings=True)}
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            conv.read_shape(cfg, None)


def f32_conversion(src: Path, dst: Path, *extra: str) -> None:
    """A conversion with every matrix stored f32 (exact values)."""
    saved = conv.tensor_type
    conv.tensor_type = lambda name, cols, recipe: "F32"
    try:
        conv.main(["--src", str(src), "--dst", str(dst), "--jobs", "1", *extra])
    finally:
        conv.tensor_type = saved


class Inert(unittest.TestCase):
    """The Preview's untrained parts: blocks whose output projections are
    ~1e-30 compute nothing and are left out; so is a parallel branch that
    is inert in every layer. A block with any live output is kept."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.src = root / "src"
        cls.src.mkdir()
        t = source_tensors(np.random.default_rng(12))
        tiny = lambda a: (np.sign(a) * 1e-30).astype(f32)  # noqa: E731
        pre = "model.language_model.layers."
        for il in range(LAYERS):
            n = f"{pre}{il}.mlp.parallel_ffn.down_proj.weight"
            t[n] = tiny(t[n])
        # Layer 1 (GDN): inert. Layer 2 (GDN): only its mixer is inert.
        for n in (f"{pre}1.delta_attn.out_proj.weight", f"{pre}1.mlp.down_proj.weight", f"{pre}2.delta_attn.out_proj.weight"):
            t[n] = tiny(t[n])
        cls.tensors = {k: bf16(v) for k, v in t.items()}
        write_checkpoint(cls.src, cls.tensors)
        f32_conversion(cls.src, root / "pruned")
        f32_conversion(cls.src, root / "kept", "--keep-inert")
        cls.pruned = root / "pruned" / "Agnes-3.0-Flash-Q4_K.gguf"
        cls.pruned_mtp = root / "pruned" / "mtp-Agnes-3.0-Flash-Q4_K.gguf"
        cls.kept = root / "kept" / "Agnes-3.0-Flash-Q4_K.gguf"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_inert_block_and_branch_are_left_out(self):
        meta, table, _ = conv.read_gguf(self.pruned)
        self.assertEqual(meta["qwen35.block_count"], LAYERS - 1)
        self.assertEqual(meta["qwen35.attention.recurrent_layers"], [True, True, False])
        # Not every fourth layer any more: no interval that would say so.
        self.assertNotIn("qwen35.full_attention_interval", meta)
        self.assertEqual(meta["qwen35.feed_forward_length"], FF)
        self.assertEqual(table["blk.0.ffn_gate.weight"][0], (H, FF))
        self.assertFalse(any(n.startswith("blk.3.") for n in table))

    def test_blocks_are_renumbered_in_order(self):
        pre = "model.language_model.layers."
        for out, il in ((0, 0), (1, 2), (2, 3)):
            got = load_gguf_tensor(self.pruned, f"blk.{out}.attn_norm.weight")
            np.testing.assert_array_equal(got, self.tensors[f"{pre}{il}.input_layernorm.weight"] + f32(1))
        got = load_gguf_tensor(self.pruned, "blk.1.ffn_down.weight")
        np.testing.assert_array_equal(got, self.tensors[f"{pre}2.mlp.down_proj.weight"])

    def test_mtp_follows_the_kept_blocks_unpadded(self):
        meta, table, _ = conv.read_gguf(self.pruned_mtp)
        self.assertEqual(meta["qwen35.block_count"], LAYERS)
        self.assertEqual(meta["qwen35.attention.recurrent_layers"], [True, True, False, False])
        self.assertEqual(table[f"blk.{LAYERS - 1}.ffn_gate.weight"][0], (H, FF))
        self.assertIn(f"blk.{LAYERS - 1}.nextn.eh_proj.weight", table)

    def test_keep_inert_keeps_and_folds_everything(self):
        meta, table, _ = conv.read_gguf(self.kept)
        self.assertEqual(meta["qwen35.block_count"], LAYERS)
        self.assertEqual(meta["qwen35.feed_forward_length"], FF + FP)
        self.assertIn("blk.3.ffn_gate.weight", table)

    def test_absmax_reads_bf16_magnitudes(self):
        src = conv.read_source(self.src)
        pre = "model.language_model.layers."
        self.assertLess(src[f"{pre}1.mlp.down_proj.weight"].absmax(), 1e-29)
        want = float(np.abs(self.tensors[f"{pre}0.mlp.down_proj.weight"]).max())
        self.assertEqual(src[f"{pre}0.mlp.down_proj.weight"].absmax(), want)


if __name__ == "__main__":
    unittest.main()
