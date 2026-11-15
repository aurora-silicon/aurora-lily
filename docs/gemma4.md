# Gemma 4 26B-A4B

lily serves Gemma 4 26B-A4B (25.2B parameters, about 3.8B active: 30
layers, 128 routed experts top 8 plus a dense MLP in every layer) from the
published GGUF files, loaded as they are, on the Linux/Vulkan backend. The
implementation follows llama.cpp's (MIT, credited in `NOTICE`), which is
also the reference lily's results are checked against.

## Files for 16 GB and 32 GB machines

The GPU gets at most `heap_memory_percent` (0.84) of the machine's memory
and the desktop needs some, so lily keeps everything it allocates under
11 GB on a 16 GB machine and under 24 GB on a 32 GB one.

| machine | file (huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | GB | context |
|---|---|---|---|
| 16 GB | `gemma-4-26B-A4B-it-UD-IQ2_M.gguf` | 10.01 | 8K |
| 32 GB | `gemma-4-26B-A4B-it-UD-Q5_K_M.gguf` | 21.15 | 32K |

The routed experts' down projections are 704 values wide, not a multiple of
256, so no file stores them as K-quants: the 2-bit files keep them at 4 bits
(IQ4_NL) and put the low bits in the gate/up experts. Measured with
llama.cpp on the same chat-shaped text (KL divergence from the Q8_0 file,
per token): UD-IQ2_M 0.158 on model-written answers and 0.298 on human text
continued in a chat turn, against 0.220/0.352 for bartowski's IQ2_S and
0.324/0.484 for IQ2_XXS; UD-Q5_K_M 0.0036/0.017. The Q8_0 file (26.8 GB)
does not fit 24 GB, nor does UD-Q6_K at 32K.

Block types: UD-IQ2_M has Q5_K, Q6_K and Q8_0 (attention, dense MLP,
embedding), IQ2_XXS and one layer of IQ2_S (routed gate/up), IQ4_NL (routed
down); UD-Q5_K_M is Q8_0 but for its Q5_K routed gate/up. Q4_0 (the draft
model below) also loads. Other Gemma 4 26B-A4B files load when every tensor
is one of the types the kernels read; Gemma 4's other sizes (per-layer
embeddings, shared KV layers) are refused.

## Running it

```sh
# 16 GB
lily --model ~/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf --max-seq 8192 \
    --cache-bytes 0.6G --pin-weights off
# 32 GB
lily --model ~/models/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf --max-seq 32768 \
    --cache-bytes 2G --pin-weights off --mtp-drafts 2
```

Run with `hk_host_memory_read_only=true hk_host_memory_import_read_only=true`:
the weights are mapped read-only from the page cache.

- `--mtp-drafts N` (N <= 3): speculative decoding with Gemma's draft model,
  the `gemma4-assistant` GGUF lily looks for beside the model
  (`mtp-gemma-4-26B-A4B-it.gguf` from the same repository, Q8_0, 0.45 GB, or
  ggml-org's `mtp-gemma-4-26B-A4B-it-Q4_0.gguf`, 0.24 GB). Drafts are greedy
  and proposed while the draft model's probability of them is at least 0.3;
  the verify pass draws under the request's sampler. The 16 GB
  configuration measures 10.9 GB without a draft model and 11.1 GB with the
  Q4_0 one: leave it off there.
- Reasoning: `chat_template_kwargs: {"enable_thinking": true}` makes the
  model think in its `<|channel>thought ... <channel|>` block, returned as
  `reasoning_content`. Tool calls (`<|tool_call>call:name{...}<tool_call|>`)
  come back as OpenAI tool calls, and tool results go back through the
  template.

Memory (lily's allocation counter, `lily-gemma4 mem`): UD-IQ2_M at 8K
context with one checkpoint 10.72 GB (weights 10.00, scratch 0.08, session
0.43, checkpoint 0.21); the server peaked at 10.9 GB after a 7.5K-token
request and its cached repeat. UD-Q5_K_M at 32K with three checkpoints 22.79
GB; the server with two drafts peaked at 23.6 GB after four cached sessions.
A session holds the 5 global layers' K/V for its whole context (20 KB a
token) and a ring of the last 1024 + 256 positions for the 25
sliding-window layers (0.26 GB); a checkpoint keeps the rings' last 1024
positions (0.21 GB).

## What runs

`src/gemma4/`: `config` (the GGUF metadata, checked), `model` (weights mapped
from the file, caches, the forward pass), `draft` (the draft model),
`engine` (the engine interface), `tokenizer` (the tokenizers JSON and
template from the GGUF). Kernels: `src/kernels/gemma4.rs`,
`vulkan/gemma4.comp` and `gemma4_gemm.comp`; the projections use the GGUF
GEMV and GEMM of `kernels/glm53` with Gemma's block types added.

A decode step is one pass: GGUF GEMVs for the projections, per-head norms
and RoPE with the K/V store, attention over 256-key chunks (two query heads
a subgroup), the router, the routed experts (a subgroup a row), and row
kernels that fuse the norms around each sublayer. A prompt runs in passes of
256 rows: GEMMs on the matrix units, the routed experts grouped by expert on
the matrix units, a row-per-subgroup attention.

`lily-gemma4` is the bring-up tool: `score` (teacher-forced NLL of token-id
cases, comparable with `tools/reference/gemma4/llama_score.c` on the same
file), `chat`, `bench`, `spec`, `state-check`, `mem`.

## Quality and speed (M5 Max, 2026-11-14)

Against llama.cpp on the same file and tokens, per-token mean NLL:
UD-Q5_K_M 0.163 (llama.cpp 0.164) on model-written chat answers, 2.956
(2.956) on human text continued in a chat turn; UD-IQ2_M 0.330 (0.334) and
2.923 (2.927). The KL divergence between lily and llama.cpp on the same
weights is about half of what Q8_0 to Q5_K costs (llama.cpp's CPU path
quantizes the activations to 8 bits; lily keeps them f32).

| | UD-IQ2_M | UD-Q5_K_M |
|---|---|---|
| prefill 1K / 4K / 16K | 1.5K tok/s at 4K, 1.0K at 16K | |
| decode (plain, 1K context) | about 80 tok/s | about 80 tok/s |
| decode, 2 drafts (prose, code) | 121, 124 tok/s | 132, 150 tok/s |

Not done: Metal kernels; prompt attention on the matrix units (a 31K
prompt prefills at about 140-700 tok/s depending on the global layers);
batching of decode steps across sessions; vision and audio towers.
