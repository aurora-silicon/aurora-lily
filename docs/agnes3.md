# Agnes 3.0 Flash

lily serves Agnes 3.0 Flash, the open-weight Preview checkpoint of
[Agnes-AI/Agnes-3.0-Flash](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash)
(Apache 2.0), as a GGUF of llama.cpp's `qwen35` architecture that
`tools/convert/convert_agnes3_flash.py` writes from the published BF16
weights. No GGUF of it is published. The model runs through the Qwen3.6
code (`src/qwen36/`), which loads both `qwen35moe` (Qwen3.6-35B-A3B) and
`qwen35` (the dense models) files; llama.cpp's `qwen35` graph is the
reference its results are checked against. The vision tower is not
converted: lily serves text.

## The model

Agnes's text decoder is Qwen3.5's dense one: Gated DeltaNet layers with a
gated full-attention layer every fourth (24 query heads over 4 KV heads of
256, partial interleaved rotary over 64 dimensions), 48 GDN value heads over
16 key heads of 128, a SwiGLU MLP, the Qwen3.5 tokenizer and chat template,
and one MTP layer. To that it adds a second, 2048-wide SwiGLU in every
layer's MLP.

The Preview is not trained everywhere (measured 2026-11-17): every layer's
second SwiGLU, and the whole of layers 20-23 and 48-51, hold initial values
(projections of RMS 0.02 with no zeros, where trained ones are 30 % zeros)
with output projections near 1e-30. They add nothing to the residual
stream. The converter leaves them out after checking that every output
projection of the part is below 1e-20, so the file is a 64-layer model with
a 17 408-wide MLP: Qwen3.5-27B's shape, 27B parameters, 3.0 GB smaller at
4 bits than the 72-layer form and as much less read a decoded token.
`--keep-inert` converts everything (the branch then folded into the main
MLP, 19 456 wide).

## Files

| file | trunk | MTP file | for |
|---|---|---|---|
| `--recipe q4_k` (default) | 15.87 GB | 2.32 GB | 32 GB and 36 GB machines |
| `--recipe q8_0` | 28.60 GB | 3.16 GB | 64 GB and up |

`q4_k` stores every matrix as Q4_K but `attn_k`, `attn_v` and the LM head
(Q8_0); `q8_0` stores every matrix as Q8_0. The MTP file (`mtp-*.gguf`)
goes beside the trunk, one conversion a directory: lily takes the first
`mtp-*.gguf` it finds there. Converting takes some minutes and 7.3 GiB of
memory whatever the machine (`tools/README.md`).

```sh
.venv/bin/python tools/convert/convert_agnes3_flash.py \
    --src ~/models/Agnes-3.0-Flash --dst ~/models/Agnes-3.0-Flash-Q4_K
```

## Running it

```sh
# 32 GB: a 77 428-token context
lily --model ~/models/Agnes-3.0-Flash-Q4_K/Agnes-3.0-Flash-Q4_K.gguf \
    --memory-limit-gb 22 --mtp-drafts 1
# 36 GB: a 123 688-token context
lily --model ~/models/Agnes-3.0-Flash-Q4_K/Agnes-3.0-Flash-Q4_K.gguf \
    --memory-limit-gb 25 --mtp-drafts 1
```

The contexts are what `--memory-limit-gb` fits after the weights and the
scratch (16.8 GB allocated, 69 632 bytes a token of bf16 K/V caches over the
16 attention layers and the MTP block); measured on the server's load line,
2026-11-17. The API serves the model as `agnes-3.0-flash` (the id comes
from the file's `general.name`). Thinking is on by default, as Agnes's
template has it, and the model thinks at length; `chat_template_kwargs:
{"enable_thinking": false}` turns it off for a request.

## Without matrix units (M4)

An M4's GPU has no matrix units lily can use and no bfloat type, so Qwen3.6's
matrix-unit kernels are refused there. The dense model has a path without
them:

- Prompt projections take the f32 GEMM (`sgemm.comp`) and the GDN its
  row-parallel scan instead of the chunked one.
- The attention kernels that use the bfloat type have `_bits` variants that
  pack and widen bfloats by hand (`common.glsl`), with the bfloat kernels'
  rounding and widening: the same caches and, by construction, the same
  results bit for bit. The tests that compare the two
  (`bits_kernels_are_bit_identical_to_their_bfloat_kernels`,
  `attn_prep_and_gate_bits_are_bit_identical`) need a device with matrix
  units, and pass on an M5 Max (2026-11-17); elsewhere they return at once.
- Decode attention takes the split kernel's variant at every length, so the
  scratch holds splits of 64 keys for the kernels' whole 262 144 (96 MiB for
  24 query heads; where the bfloat kernels run, contexts past 8 192 keys
  take other routes and the scratch is 2 MB).
- Prompt attention is `sdpa_prefill_rows`: one subgroup a row and query
  head, a key head's group of query heads in one threadgroup, an online
  softmax over the keys.

This path is verified on a Radeon 8060S (RADV, which has no bfloat type
either: lily treats it as a GPU without matrix units), not yet on an M4.
There, with the `q4_k` file (2026-11-17): a 2 048-token prompt at 140-150
tokens a second, 13.2 tokens a second decoded plain and 18-21 with one MTP
draft a step (78-95 % of drafts accepted). These are not Apple numbers.

## Speed on an M5 Max

`lily-qwen36 bench` (a 4 096-token prompt, then 64 tokens decoded) and
`lily-qwen36 spec` (128 tokens with one MTP draft a step, against plain
greedy decoding), two interleaved runs a file, 16-inch M5 Max, 2026-11-17:

| file | prefill | decode, plain | decode with one draft |
|---|---|---|---|
| `q4_k` | 700-740 tok/s | 24.6-25.3 tok/s | 31.0 tok/s (84 % accepted) |
| `q8_0` | 680-740 tok/s | 17.1-17.8 tok/s | 26.2 tok/s (85 % accepted) |

A dense 27B model reads all its weights every token, so decode is bounded
by memory bandwidth (the `q4_k` file is 15.9 GB). Served, a short code
answer decoded at 36 tokens a second with 47 of 49 drafts accepted.

## Checks

Against llama.cpp (CPU backend, the same file), teacher-forced, on the
Radeon 8060S path, 2026-11-17:

| file | text | argmax agreement | NLL per token, lily / llama.cpp |
|---|---|---|---|
| `q4_k`, 64 layers | 220 tokens of `CONTRIBUTING.md`, the last 100 scored | 97/100 | 3.0819 / 3.0812 |
| `q4_k --layers 4` | a 42-token Python function, 32 scored | 32/32 | 7.7981 / 7.7961 |
| `q4_k --layers 4` | 9 000 tokens of `README.md`, the last 100 scored | 100/100 | 7.8892 / 7.8883 |
| `q4_k`, M5 Max (matrix units) | the 220 tokens above | 96/100 | 3.0835 / 3.0812 |
| `q4_k --layers 4`, M5 Max | the 9 000 tokens above | 100/100 | 7.8894 / 7.8883 |

The disagreements are near-ties in llama.cpp's own logits (margins 0.048
to 0.171); the M5's matrix-unit path rounds the prompt GEMMs' operands to
bfloat, which is its NLL's extra 0.002 (Qwen3.6's chunked prefill added
0.001 to 0.003 the same way). On the M5 the branch leaves Qwen3.6's output
unchanged byte for byte (teacher-forced NLL, a 9 000-token context and
greedy generation, against `main`). Speculative decoding with the MTP head gives the
same tokens as plain greedy decoding (64 of 64; `lily-qwen36 spec`).
