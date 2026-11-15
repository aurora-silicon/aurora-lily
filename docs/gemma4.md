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
| 16 GB | `gemma-4-26B-A4B-it-UD-IQ2_M.gguf` | 10.01 | 22.7K (7.9K with the draft model) |
| 32 GB | `gemma-4-26B-A4B-it-UD-Q5_K_M.gguf` | 21.15 | 45K |

The contexts are what `--memory-limit-gb` fits with the flags below.

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
# 16 GB: everything under 11.0 GB, a 22.7K-token context
lily --model ~/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf --memory-limit-gb 10.2 \
    --mtp-drafts 0 --checkpoints-per-session 1 --decode-checkpoint-tokens 0 \
    --pin-weights off
# 16 GB with ggml-org's Q4_0 draft model beside the file: a 7.9K-token
# context, faster on code and English (below)
lily --model ~/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf --memory-limit-gb 10.2 \
    --mtp-drafts 2 --checkpoints-per-session 1 --decode-checkpoint-tokens 0 \
    --pin-weights off
# 32 GB: under 23.6 GB, a 45K-token context, two drafts a step
lily --model ~/models/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf --memory-limit-gb 22 \
    --mtp-drafts 2 --decode-checkpoint-tokens 0 --pin-weights off
```

Run with `hk_host_memory_read_only=true hk_host_memory_import_read_only=true`:
the weights are mapped read-only from the page cache.

- `--memory-limit-gb` (GiB): at load lily measures the weights and the
  scratch and fits the longest context whose session at its largest (the
  caches at the full context, the session's snapshots, growth) fits the
  rest; the server caps `--max-seq` there, answers a longer prompt with a
  400, keeps that session's room free in its cache and serves one request
  at a time. It also picks the prompt run (1024, 512 or 256 tokens; the
  scratch grows about 0.3 MB a run token, and longer runs prefill faster):
  the longest that still fits 32K tokens of context, or as many as
  256-token runs fit when that is less (the 16 GB tier keeps 256, the
  32 GB one takes 1024). Without a limit runs are 1024 tokens. Gemma's
  snapshots are large (the sliding-window layers' last
  1024 positions: 0.21 GB each), so the tiers keep fewer of them:
  `--checkpoints-per-session` (prompt-end checkpoints, 3 by default) and
  `--decode-checkpoint-tokens 0` (no checkpoints inside answers, 4 by
  default). With the defaults the same limit fits 12K tokens on 32 GB,
  and the 16 GB one refuses to load (less than 4K tokens).
- `--mtp-drafts N` (N <= 3): speculative decoding with Gemma's draft model,
  the `gemma4-assistant` GGUF lily looks for beside the model
  (`mtp-gemma-4-26B-A4B-it.gguf` from the same repository, Q8_0, 0.45 GB, or
  ggml-org's `mtp-gemma-4-26B-A4B-it-Q4_0.gguf`, 0.24 GB). Drafts are greedy
  and proposed while the draft model's probability of them is at least 0.3;
  the verify pass draws under the request's sampler. The default is on;
  on 16 GB the Q4_0 draft model costs 15K tokens of context (22.7K to 7.9K),
  so both configurations are listed above.
- Reasoning: `chat_template_kwargs: {"enable_thinking": true}` makes the
  model think in its `<|channel>thought ... <channel|>` block, returned as
  `reasoning_content`. Tool calls (`<|tool_call>call:name{...}<tool_call|>`)
  come back as OpenAI tool calls, and tool results go back through the
  template.

Memory with the two configurations above (lily's allocation counter; the
load's fit, printed at start): on 16 GB, 10.1 GB of weights and scratch and
at most 0.82 GB for the request's session (its caches at the full 22.7K
context, its checkpoint, the layer its growth holds twice), which is the
10.95 GB limit. On 32 GB, 22.0 GB of weights, the draft model and scratch
(1024-token prompt runs) and at most 1.58 GB for the session (45K tokens,
three checkpoints), the 23.6 GB limit. A request drops the checkpoints its
release would drop before it takes a new one, so a session never holds
more than the store keeps. A session holds the 5 global layers' K/V for
its whole context (12.8 KB a token) and a ring of the last 1024 + 256
positions for the 25 sliding-window layers (0.26 GB); a checkpoint keeps
the rings' last 1024 positions (0.21 GB). In the global layers V is the K
projection (rmsnorm(k)) and K is rope(V k_norm) turning only the first 64
of 256 pairs, so their K caches keep only those pairs' values and
attention takes the rest from V (times k_norm).

## What runs

`src/gemma4/`: `config` (the GGUF metadata, checked), `model` (weights mapped
from the file, caches, the forward pass), `draft` (the draft model),
`engine` (the engine interface), `tokenizer` (the tokenizers JSON and
template from the GGUF). Kernels: `src/kernels/gemma4.rs`,
`vulkan/gemma4.comp` and `gemma4_gemm.comp`; the projections use the GGUF
GEMV and GEMM of `kernels/glm53` with Gemma's block types added.

A decode step is one pass: GGUF GEMVs for the projections, per-head norms
and RoPE with the K/V store, attention over 256-key chunks (two query heads
a subgroup, four keys a step), the router, the routed experts (a subgroup a
row), and row kernels that fuse the norms around each sublayer. A prompt
runs in passes of up to 1024 rows: bfloat-input GEMMs on the matrix units,
the routed experts grouped by expert on the matrix units, and flash
attention on the matrix units (`gemma4_attn.comp`: 16 rows of a KV head's
query heads a threadgroup, f16 operands, f32 sums), 256 rows at a time.

`lily-gemma4` is the bring-up tool: `score` (teacher-forced NLL of token-id
cases, comparable with `tools/reference/gemma4/llama_score.c` on the same
file), `chat`, `bench`, `spec`, `state-check`, `mem`, `fit` (the context a
`--memory-limit-gb` fits).

## Quality and speed (M5 Max, 2026-11-15)

Against llama.cpp on the same file and tokens, per-token mean NLL:
UD-Q5_K_M 0.164 (llama.cpp 0.164) on model-written chat answers, 2.955
(2.956) on human text continued in a chat turn; UD-IQ2_M 0.330 (0.334) and
2.918 (2.927). The KL divergence from llama.cpp to lily on the same weights
is 0.0018 / 0.010 on UD-Q5_K_M, about half of what Q8_0 to UD-Q5_K_M costs
(0.0036 / 0.017), and 0.0077 / 0.026 on UD-IQ2_M, 5% and 9% of what Q8_0 to
UD-IQ2_M costs (0.158 / 0.298). llama.cpp's CPU path quantizes the
activations to 8 bits; lily keeps them f32.

`lily-gemma4 bench` (greedy, English text, 1024-token prompt runs) and
`spec` (2 drafts, greedy, 256 tokens), Mesa `mesa-night`:

| | UD-IQ2_M | UD-Q5_K_M |
|---|---|---|
| prefill 1K / 4K / 16K tokens | 3920 / 3750 / 3310 tok/s | 3560 / 3440 / 3060 tok/s |
| prefill 31K | | 2710 tok/s |
| decode, plain (1K / 4K / 16K context) | 8.3 / 8.5 / 9.3 ms a step | 9.8 / 9.9 / 10.9 ms |
| decode, 2 drafts (code, prose, Italian) | 153, 133, 112 tok/s (Q4_0 draft) | 164, 134, 122 tok/s |

Served (the standard five requests, 400-token answers): the 16 GB
configuration decodes at 124 tok/s (116 after the 4.9K-token prompt, which
prefills at 2520 tok/s); the 32 GB one at 140 to 164 tok/s with 66 to 84% of
the drafts accepted, its 4.9K-token prompt at 3270 tok/s.

Not done: Metal kernels; batching of decode steps across sessions; vision
and audio towers.
