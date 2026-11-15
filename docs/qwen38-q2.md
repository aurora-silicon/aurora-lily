# Qwen3.8-Flash-Next from ds4's Q2 GGUF (the 64 GB tier)

lily serves Qwen3.8-Flash-Next from antirez's `qwen38-q2` GGUF
(`antirez/qwen3.8-flash-next-gguf`, `Qwen3.8-Flash-Next-Q2.gguf`, 147 207 127 040
bytes): IQ2_XXS gate/up and Q2_K down routed experts, Q8_0 projections, F16
hyper-connection mixers, the MTP block with Q4_K/MXFP4 experts. It runs the same
graph as lily's own checkpoints (`src/qwen4exp/`), on GGUF blocks read in place
(`src/qwen4exp/gguf.rs`, `src/kernels/qwen_gguf.rs`,
`src/kernels/vulkan/qwen_gguf.comp`, `qwen_gguf_gemm.comp`).

## The file

Everything but the n-gram table is the first 44.8 GB of the file (41.7 GiB; the
draft head is 1.4 GB of it); the BF16 n-gram table (`per_layer_token_embd`, 95 GiB)
is the rest. lily does not read that table yet: it pages the same table, at 4 bits,
from one of its own Qwen3.8-Flash-Next checkpoints (`--ngram-checkpoint <dir>`,
`LILY_NGRAM_TABLE`), whose hashing constants must equal the file's. On a sampled
comparison the 4-bit rows have a cosine of 0.996-0.997 to the BF16 rows, the
quantization error alone. The main weights can be fetched alone with HTTP range
requests into a sparse file of the full size (the offsets stay valid).

The loader undoes the converter's layout choices with exact moves (llama.cpp's
`conversion/qwen4exp.py` and `qwen.py`, which ds4's converter wraps):

- GDN value heads from tiled back to grouped order (v rows of `attn_qkv`,
  `attn_gate`, `ssm_alpha`/`ssm_beta`, `ssm_a`, `ssm_dt.bias`, the v channels of
  `ssm_conv1d`, the input columns of `ssm_out`, which are owned copies);
- folded norms `1 + w` back to bfloat16 `w` (exact per value, checked);
- `ssm_a = -exp(A_log)` back to `A_log`; conv weights to tap-major;
- the F32 routers, shared gates and alpha/beta, which hold bfloat16 values, as
  BF16 (checked per value);
- the draft head's `eh_proj` as its `fc_embedding` and `fc_hidden` column views;
- the routed down experts' rows are padded from 640 to 768 values: the kernels read
  640.

## Running it

```sh
./target/release/lily --model ~/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --ngram-checkpoint ~/models/Qwen3.8-Flash-Next-lily-q4-xl \
  --memory-limit-gb 44.7
```

Memory: `--memory-limit-gb` (GiB; 44.7 GiB is 48.0 GB, the 64 GB tier's GPU
budget) is the footprint the server keeps to. The weights take 44.8 GB of it; the
rest holds the scratch and one session. The load measures the weights and a scratch
with its full prefill chunk (allocated from the start) and fits the longest context
whose session at its largest fits the rest: its caches grown to the context and the
speculative lookahead, the server's checkpoints, and the one attention layer that
growth holds twice. The server caps `--max-seq` there and refuses longer prompts.
Under the limit it serves one request at a time (no batched decoding, whose scratch
the fit leaves out), and its session cache keeps that largest session free before
each request allocates: the other sessions go to the disk tier first, and a fork
that would not fit beside its source rewinds the source instead. Where a 4096-row
prefill chunk leaves less than 32K tokens the chunk halves (about 5% slower prefill
at 2048). With the draft head, q8 K/V:

| `--memory-limit-gb` | context | prefill chunk | allocated | session cache (kept free) |
|---|---:|---:|---:|---:|
| 44.7 (48.0 GB) | 32 760 | 2048 | 46.3 GB | 1.7 GB (1.57 GB) |
| 48 (51.5 GB) | 163 832 | 4096 | 47.3 GB | 4.2 GB (4.16 GB) |

Persisted sessions (the disk tier) are keyed by the file's identity: a hash of its
header and tensor table, sampled blocks of its tensor data, and the n-gram
checkpoint's identity. Another quantization or another n-gram table never resumes
them.

The n-gram table's background preload is skipped where the page cache cannot hold
it beside the GPU's allocation.

## Quality and speed (M5 Max, 2026-11-14)

- ds4's official continuations (`gguf-tools/quality-testing`, ds4's no-thinking
  prompt, `lily-qwen38 score`): 100 short cases NLL 0.3537, first token 68/100 (ds4's
  CUDA run of the same file: 0.3530, 70/100); 12 long cases (2K-24K prompts)
  0.1536, 8/12 (ds4: 0.1542, 7/12).
- llama.cpp 0.6.0 on the same file (patched to accept it) and the same n-gram rows,
  per-token NLL of 24 raw C/English/Italian continuations: within 0.011 per language
  (`lily-qwen38 nll`).
- Prefill 1K / 4K / 8K: ~1610 / ~1900 / ~1760 tok/s; decode ~54-55 tok/s; with the
  MTP draft head ~61-79 tok/s at one draft (prose to code), up to ~91 at two on code.

## Not done

- The file's own BF16 n-gram table (a complete ds4 download still needs a lily
  checkpoint's table).
- The token embedding (1.27 GB) stays on the GPU; paging it from the file would
  lower the footprint further.
- Vision (ds4's separate mmproj file).
