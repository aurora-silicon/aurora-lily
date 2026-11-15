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
  --max-seq 32768 --memory-limit-gb 48
```

Memory (the server's line, 32K context, q8 K/V): 43.4 GB allocated without the
draft head (`--mtp-drafts 0`), 45.3 GB with it; `--memory-limit-gb 48` (GiB) leaves
a 6.3 GB session cache. The n-gram table's background preload is skipped where the
page cache cannot hold it beside the GPU's allocation.

## Quality and speed (M5 Max, 2026-11-14)

- ds4's official continuations (`gguf-tools/quality-testing`, ds4's no-thinking
  prompt, `lily-qwen score`): 100 short cases NLL 0.3537, first token 68/100 (ds4's
  CUDA run of the same file: 0.3530, 70/100); 12 long cases (2K-24K prompts)
  0.1536, 8/12 (ds4: 0.1542, 7/12).
- llama.cpp 0.6.0 on the same file (patched to accept it) and the same n-gram rows,
  per-token NLL of 24 raw C/English/Italian continuations: within 0.011 per language
  (`lily-qwen nll`).
- Prefill 1K / 4K / 8K: ~1610 / ~1900 / ~1760 tok/s; decode ~54-55 tok/s; with the
  MTP draft head ~61-79 tok/s at one draft (prose to code), up to ~91 at two on code.

## Not done

- The file's own BF16 n-gram table (a complete ds4 download still needs a lily
  checkpoint's table).
- The token embedding (1.27 GB) stays on the GPU; paging it from the file would
  lower the footprint further.
- Vision (ds4's separate mmproj file).
