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
| 16 GB | `gemma-4-26B-A4B-it-UD-IQ2_M.gguf` | 10.01 | 14.5K |
| 32 GB | `gemma-4-26B-A4B-it-UD-Q5_K_M.gguf` | 21.15 | 41K |

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
# 16 GB: everything under 11.0 GB, a 14.5K-token context
lily --model ~/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf --memory-limit-gb 10.2 \
    --mtp-drafts 0 --checkpoints-per-session 1 --decode-checkpoint-tokens 0 \
    --pin-weights off
# 32 GB: under 23.6 GB, a 41K-token context, two drafts a step
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
  at a time. Gemma's snapshots are large (the sliding-window layers' last
  1024 positions: 0.21 GB each), so the tiers keep fewer of them:
  `--checkpoints-per-session` (prompt-end checkpoints, 3 by default) and
  `--decode-checkpoint-tokens 0` (no checkpoints inside answers, 4 by
  default). With the defaults the same limit fits under 10K tokens on 32 GB,
  and the 16 GB one refuses to load (less than 4K tokens).
- `--mtp-drafts N` (N <= 3): speculative decoding with Gemma's draft model,
  the `gemma4-assistant` GGUF lily looks for beside the model
  (`mtp-gemma-4-26B-A4B-it.gguf` from the same repository, Q8_0, 0.45 GB, or
  ggml-org's `mtp-gemma-4-26B-A4B-it-Q4_0.gguf`, 0.24 GB). Drafts are greedy
  and proposed while the draft model's probability of them is at least 0.3;
  the verify pass draws under the request's sampler. The default is on;
  the 16 GB configuration leaves it off (the Q4_0 draft model would cost it
  9K tokens of context).
- Reasoning: `chat_template_kwargs: {"enable_thinking": true}` makes the
  model think in its `<|channel>thought ... <channel|>` block, returned as
  `reasoning_content`. Tool calls (`<|tool_call>call:name{...}<tool_call|>`)
  come back as OpenAI tool calls, and tool results go back through the
  template.

Memory (lily's allocation counter, `lily-gemma4 mem`): UD-IQ2_M at the
16 GB tier's full context with its checkpoint 10.95 GB (weights 10.00,
scratch 0.12, session 0.62, checkpoint 0.21); the server held 10.8 GB after
7.5K- and 5.8K-token requests. UD-Q5_K_M at 49K with three checkpoints
23.17 GB plus the 0.45 GB draft model; the server held 22.2 GB after a
31K-token request, its cached repeat and four short ones. A session holds
the 5 global layers' K/V for its whole context (20 KB a token) and a ring of
the last 1024 + 256 positions for the 25 sliding-window layers (0.26 GB); a
checkpoint keeps the rings' last 1024 positions (0.21 GB).

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
UD-Q5_K_M 0.164 (llama.cpp 0.164) on model-written chat answers, 2.955
(2.956) on human text continued in a chat turn; UD-IQ2_M 0.330 (0.334) and
2.923 (2.927). The KL divergence between lily and llama.cpp on the same
weights is about half of what Q8_0 to Q5_K costs (llama.cpp's CPU path
quantizes the activations to 8 bits; lily keeps them f32).

| | UD-IQ2_M | UD-Q5_K_M |
|---|---|---|
| prefill 1K / 4K tokens | 1820 / 1500 tok/s | 1343 / 1174 tok/s |
| prefill 16K | 1038 tok/s | |
| decode, plain (1K / 4K context) | 10.9 / 10.6 ms a step | 11.1 / 11.1 ms |
| decode, 2 drafts, greedy (prose, code) | 121, 124 tok/s | 131, 150 tok/s |

Served (the 32 GB configuration, 400-token answers): 115 to 145 tok/s
depending on the prompt, 58 to 87% of the drafts accepted; a 31K-token
prompt prefills at 635 tok/s and decodes at 80 tok/s. The 16 GB
configuration: a 7.5K-token prompt at 1290 tok/s, decoding at 91 tok/s.

Not done: Metal kernels; prompt attention on the matrix units (it is a third of a 4K
prompt's time and grows with the global layers' context);
batching of decode steps across sessions; vision and audio towers.
