# lily on Linux

Fast local inference of mixture-of-experts language models on Apple Silicon
running Linux. lily is a compact inference server with hand-written GPU
kernels; this fork runs it on Linux through Vulkan (Mesa's Honeykrisp
driver) and serves five model families, from a 16 GB laptop to a 128 GB
workstation. The server is OpenAI-compatible (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, tools, `reasoning_content`, streaming) and
is used daily as a coding-agent backend.

| memory | model | weights | speculative drafts |
|---|---|---|---|
| 16 GB | Gemma 4 26B-A4B | unsloth's UD-IQ2_M GGUF | optional: Gemma's draft model |
| 32 GB | Qwen3.6-35B-A3B | ggml-org's Q4_K_M GGUF | its MTP head |
| 32 GB | Gemma 4 26B-A4B | unsloth's UD-Q5_K_M GGUF | Gemma's draft model |
| 64 GB | Qwen3.6-35B-A3B | ggml-org's Q8_0 GGUF | its MTP head |
| 64 GB | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | antirez's Q2 GGUF | its MTP head |
| 128 GB | Qwen3.8-Flash-Next | lily's 4-bit conversions (q4-xl, q4) | its MTP head |
| 128 GB | GLM 5.3 Flash | antirez's Q2 GGUF | its MTP block |
| 128 GB | DeepSeek V4 Flash | antirez's Q2 GGUF | DSpark |

GGUF files load as published, without re-quantization. On the smaller
machines `--memory-limit-gb` keeps everything lily allocates on the GPU
(weights, scratch and the active session) within the budget the machine
leaves the GPU, and fits the longest context that budget allows.

## Hardware

- **M5 family** (GPUs with matrix units): every model above. Developed and
  measured on a 16-inch M5 Max MacBook Pro with 128 GB. The base M5 and M5
  Pro use the same kernels; their GPUs and memory bandwidth are smaller, so
  expect lower numbers than the tables below.
- **M4 family** (no matrix units): Gemma 4 runs, with f32 GEMMs and decode
  kernels tuned for the M4's GPU. The other models' kernels need matrix
  units and are refused on these GPUs.

## Speed

### Linux, M5 Max (128 GB)

The server as each configuration runs it (the commands below), five chat
requests, greedy, 400 generated tokens each: four short coding-agent
questions (40 to 100 prompt tokens) and one long-context request (a
14 000-character source file to summarize, 4 200 to 4 900 prompt tokens).
Measured 2026-11-15 on `main` with Mesa `aurora-wip` plus the AGX compiler
pull requests aurora-silicon/mesa #370 to #375, tokens per second, on an
otherwise idle machine:

| memory | model | decode | drafts accepted | prefill, long prompt | prefill, short prompts |
|---|---|---:|---:|---:|---:|
| 16 GB | Gemma 4 UD-IQ2_M | 118-128 | (no drafts) | 3 390 | 820-1 000 |
| 32 GB | Qwen3.6 Q4_K_M | 140-148 | 87-90% | 3 550 | 530-920 |
| 32 GB | Gemma 4 UD-Q5_K_M | 140-161 | 68-83% | 3 540 | 680-920 |
| 64 GB | Qwen3.6 Q8_0 | 138-146 | 86-89% | 3 260 | 640-820 |
| 64 GB | Qwen3.8-Flash-Next Q2 | 80-98 | 62-89% | 1 580 | 200-340 |
| 128 GB | Qwen3.8-Flash-Next q4-xl | 104-124 | 68-88% | 1 770 | 460-530 |
| 128 GB | DeepSeek V4 Flash Q2 | 50-56 | 68-81% | 830 | 145-185 |
| 128 GB | GLM 5.3 Flash Q2 | 41-47 | 58-80% | 580 | 195-220 |

Short prompts mostly measure fixed per-request costs. Agent output (tool
calls, code copied from the context) drafts well, so decode in agent use is
often faster than this set. Without the Mesa compiler pull requests, prefill
is 2 to 15% lower (Gemma loses the most) and decode 0 to 3% lower.

### Compared with macOS

The same M5 Max model under macOS, as published by the projects lily on
Linux follows:

| model | engine on macOS (Metal) | prefill | decode | lily on Linux |
|---|---|---:|---:|---|
| Qwen3.8-Flash-Next q4-xl | [fabiogreter's lily](https://github.com/fabiogreter/lily-qwen3.8-flash-next) (`docs/performance.md`, 2026-10-12: fresh prompts over HTTP, 4K-64K) | 2 410-2 524 | 98-107 with 2 drafts, 77-80 plain | prefill 1 770, decode 104-124 with 2 drafts, ~69 plain |
| DeepSeek V4 Flash Q2 | [antirez's ds4](https://github.com/antirez/ds4) (`speed-bench/m5_max.csv`) | 790 at 2K, 710 at 4K | 39 plain | prefill ~1 000 at 2K, 830 at 4.4K; decode ~40 plain, 50-56 with DSpark |

Decode on Linux matches or beats macOS; Qwen3.8's prefill is still about
30% behind lily on Metal. Neither project publishes M5 Max numbers for the
other models.

### Linux, M4 (16 GB)

Gemma 4 UD-IQ2_M on an M4 MacBook Air (8-core GPU, fanless, measured below
60 °C): decode about 25 tok/s (a 40 ms step at 1K context), prefill 160 to
190 tok/s at 1K and 132 at 4K, 21 760 tokens of context at
`--memory-limit-gb 10.2`.

### Quality

| model | check | lily | reference |
|---|---|---|---|
| Qwen3.8-Flash-Next Q2 | ds4's 100 official continuations, NLL / first tokens | 0.354, 68/100 | ds4 0.353, 70/100 |
| GLM 5.3 Flash Q2 | ds4's 100 official continuations | 0.464, 91/100 | ds4 0.458-0.462, 89-90/100 |
| DeepSeek V4 Flash Q2 | ds4's 100 official continuations | 0.399, 55/100 | ds4 0.398, 56/100 |
| Qwen3.6-35B-A3B | 24 long texts, NLL per token, against llama.cpp on the same file | Q4_K_M 1.813, Q8_0 1.809 | llama.cpp within 0.002 |
| Gemma 4 UD-Q5_K_M | KL divergence from llama.cpp on the same file (chat / human text) | 0.0018 / 0.0105 | |
| Gemma 4 UD-IQ2_M | KL divergence from llama.cpp on the same file | 0.0077 / 0.0261 | |

Speculative output equals plain greedy decoding up to rounding ties.

### Memory

| memory | configuration | everything on the GPU | longest context |
|---|---|---:|---:|
| 16 GB | Gemma 4 UD-IQ2_M, `--memory-limit-gb 10.2` | 11.0 GB | 21 248 (10 496 with the draft model) |
| 32 GB | Qwen3.6 Q4_K_M, `--memory-limit-gb 22` | 23.6 GB | 52 088 |
| 32 GB | Gemma 4 UD-Q5_K_M, `--memory-limit-gb 22` | 23.6 GB | 55 296 |
| 64 GB | Qwen3.6 Q8_0, `--memory-limit-gb 44` | 47.2 GB | 262 144 |
| 64 GB | Qwen3.8-Flash-Next Q2, `--memory-limit-gb 44.7` | 48.0 GB | 32 760 |
| 128 GB | Qwen3.8-Flash-Next q4-xl | 75 GB, plus the 32 GB n-gram table paged | 131 072 |
| 128 GB | GLM 5.3 Flash Q2 | 100 GB | 65 536 |
| 128 GB | DeepSeek V4 Flash Q2 | 95 GB | 65 536 |

`--memory-limit-gb` counts GiB. A memory-limited configuration serves one
request at a time.

## Running it

You need:

- Linux on Apple Silicon with the [aurora-silicon](https://github.com/aurora-silicon)
  kernel (`integration/aurora`), including the GPU compute-time allowance
  (aurora-silicon/linux #1752); on the M5 Max, the GPU mailbox interrupt fix
  (#1844) for full decode speed.
- Mesa's Honeykrisp from aurora-silicon/mesa `aurora-wip`, with the M5's
  matrix units enabled (`ASAHI_M5_COOPERATIVE_MATRIX=1`). The AGX compiler
  pull requests #370 to #375 add the prefill gains measured above.
- The Rust toolchain pinned by `rust-toolchain.toml`.

Weights are mapped read-only from the page cache rather than copied, and the
GPU heap needs more than half of RAM:

```sh
export VK_ICD_FILENAMES=$MESA/share/vulkan/icd.d/asahi_icd.aarch64.json
export LD_LIBRARY_PATH=$MESA/lib
export ASAHI_M5_COOPERATIVE_MATRIX=1 heap_memory_percent=0.84
export hk_host_memory_read_only=true hk_host_memory_import_read_only=true

cargo build --release --locked
```

Then one of (a `.gguf` path selects the model from its metadata; draft
models and support files are found beside the model):

```sh
# 16 GB: Gemma 4, everything under 11 GB (add --mtp-drafts 2 with the draft model)
./target/release/lily --model ~/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf \
  --memory-limit-gb 10.2 --mtp-drafts 0 --checkpoints-per-session 1 \
  --decode-checkpoint-tokens 0 --pin-weights off

# 32 GB: Qwen3.6 Q4_K_M, or Gemma 4 UD-Q5_K_M with
#   --memory-limit-gb 22 --mtp-drafts 2 --decode-checkpoint-tokens 0 --pin-weights off
./target/release/lily --model ~/models/Qwen3.6-35B-A3B-Q4_K_M.gguf --memory-limit-gb 22

# 64 GB: Qwen3.6 Q8_0 (--memory-limit-gb 44), or Qwen3.8-Flash-Next Q2, whose
# n-gram table is paged from a lily checkpoint of the model (docs/qwen38-q2.md)
./target/release/lily --model ~/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --ngram-checkpoint ~/models/Qwen3.8-Flash-Next-lily-q4-xl --memory-limit-gb 44.7

# 128 GB: Qwen3.8-Flash-Next q4-xl
./target/release/lily --model ~/models/Qwen3.8-Flash-Next-lily-q4-xl --max-seq 131072

# 128 GB: GLM 5.3 Flash
./target/release/lily --model ~/models/GLM-5.3-Flash-Q2/GLM-5.3-Flash-Q2.gguf \
  --max-seq 65536 --cache-bytes 9G --pin-weights off --mtp-drafts 1 --max-batch 1

# 128 GB: DeepSeek V4 Flash with DSpark
./target/release/lily --model ~/models/DeepSeek-V4-Flash-Q2/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --max-seq 65536 --cache-bytes 9G --pin-weights off --mtp-drafts 5 --max-batch 1
```

The server listens on `127.0.0.1:8000` unless `--bind` says otherwise. Only
one model fits in 128 GB at a time; every loader takes a machine-wide
instance lock before mapping weights.

## The models

- **Qwen3.8-Flash-Next**: Qwen's `qwen4_exp` preview: 48 layers (36 Gated
  DeltaNet, 12 sparse attention with an indexer), 512 experts with 10
  active, a four-stream gated residual, a 32 GB hashed n-gram embedding and
  a draft head. Two 4-bit conversions,
  **[q4-xl](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl)**
  (recommended, 106.9 GB; attention, shared expert, LM head and embedding at
  8 bits) and **[q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4)**
  (105.5 GB), made by `tools/convert/convert_qwen38_flash_next.py`
  ([format](docs/qwen38-flash-next-checkpoint-format.md)), and antirez's Q2
  GGUF for 64 GB machines ([docs/qwen38-q2.md](docs/qwen38-q2.md)).
- **Qwen3.6-35B-A3B**: Gated DeltaNet and gated attention at 3:1, 256
  experts with 8 active and a gated shared expert, an MTP head loaded from
  ggml-org's separate MTP file.
- **Gemma 4 26B-A4B**: 25 sliding-window and 5 global attention layers, a
  dense GeGLU MLP and 128 experts with 8 active in every layer; drafts from
  Gemma's own draft model, which reads the target's caches
  ([docs/gemma4.md](docs/gemma4.md)).
- **GLM 5.3 Flash**: KDA linear attention and DSA sparse attention at 3:1,
  288 experts with one shared, four hyper-connection streams and a built-in
  MTP block ([docs/glm53.md](docs/glm53.md)).
- **DeepSeek V4 Flash**: multi-query attention over a 128-row window plus
  compressed rows (ratio 4 with a sparse indexer, or 128), FP8 KV rounding,
  256 experts with 6 active and a shared one, four hyper-connection streams;
  drafts from DSpark, antirez's three-stage block drafter
  ([docs/deepseek4.md](docs/deepseek4.md)).

Every GGUF is checked against a schema of its tensors before anything is
mapped, and every model has a CPU reference of its operations for
layer-by-layer comparison.

## How the Linux port works

`src/vulkan.rs` implements the Metal backend's API over Vulkan, so the
engine and its dispatch code are shared, and every kernel is GLSL in
`src/kernels/vulkan/` ([PORTING.md](src/kernels/vulkan/PORTING.md)).

- **Matrix units.** On the M5, prompt GEMMs and attention run on the GPU's
  matrix units through Vulkan's cooperative-matrix extensions. On GPUs
  without them (the M4) the same work runs as f32 GEMMs that decode the
  quantized weights into threadgroup memory.
- **Kernels shaped for the GPU.** A branch waits for every load in flight,
  registers bound how many loads stay in flight, dispatches in one
  dependency level do not overlap, and each dispatch fetches its code cold:
  hot kernels are specialized for the model's fixed shapes, kept small, and
  same-level work is fused.
- **Weights from the page cache.** Checkpoints are imported read-only
  (`VK_EXT_external_memory_host`), not copied.
- **The driver and kernel underneath.** Bringing lily up found and fixed
  problems below it, filed to aurora-silicon's Mesa and Linux trees: a stall
  where one client's long compute work starved other clients, barrier and
  compiler costs in the AGX3 backend, bfloat16 rounding, and a GPU mailbox
  that had fallen back to polling.

## Server features

- **Session cache.** Each conversation's state stays resident within a GPU
  memory budget; evicted sessions move to disk and survive restarts; shared
  agent preambles are stored once ([docs/architecture.md](docs/architecture.md)).
- **Batching.** Up to `--max-batch` requests decode together on Qwen3.8
  q4-xl; the GGUF models serve one request at a time.
- **Thinking controls.** A thinking budget with nudges; a tool call drawn
  inside the reasoning ends it ([docs/server.md](docs/server.md)).
- **Timings.** Every response carries prefill and decode rates, cached
  tokens and draft acceptance; `tools/opencode-plugin-timings/` shows them in
  opencode.

## Tests

```sh
cargo test --locked
```

runs the kernel tests against CPU references, the shader compilation test
and the API tests on the local GPU; they need no checkpoint. Tests that need
weights are ignored by default and take `LILY_MODEL_DIR_FLASH` (Qwen3.8),
`LILY_QWEN36_GGUF`, `LILY_GLM53_GGUF`, `LILY_DS4_GGUF` or `LILY_GEMMA4_GGUF`.
[CONTRIBUTING.md](CONTRIBUTING.md) has the gates.

## Where it comes from

- **[Perplexity's lily](https://github.com/perplexityai/pplx-garden/tree/main/lily)**
  ([write-up](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)):
  the kernel foundations (4-bit weight streaming adapted from MLX, tensor-op
  GEMMs, the MoE, Gated DeltaNet and attention kernels), the runtime shader
  compiler and the server.
- **[fabiogreter/lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next)**:
  the Qwen3.8-Flash-Next graph and converter, speculative decoding with the
  model's draft head, the session cache, and the kernel tuning the Vulkan
  kernels follow. The Qwen3.8 checkpoints are this work.
- **[antirez/ds4](https://github.com/antirez/ds4)** (MIT): the reference
  implementation GLM 5.3 Flash and DeepSeek V4 Flash follow, the quantized
  GGUFs lily loads, and the official continuation fixtures lily is scored on.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT): the graphs
  Qwen3.6 and Gemma 4 follow (and Gemma's draft loop), and the reference
  their scores are checked against; **ggml** for the GGUF block formats
  (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, IQ2_XXS, IQ2_S, IQ4_NL, MXFP4), which
  lily's decoders reproduce bit for bit.

## License

Apache-2.0. `NOTICE` records the upstream copyrights and the MLX, MLX-LM,
ds4 and llama.cpp/ggml (MIT) code adapted here. The models are under their
own licences, stated on their checkpoints' cards.
