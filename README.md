# lily on Linux: mixture-of-experts models on the Apple M5 Max, from 16 GB to 128 GB

This fork runs [lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
a compact inference server with hand-written GPU kernels, on **Linux on an
Apple M5 Max**, through Vulkan, and serves several mixture-of-experts model
families, with at least one for each memory size:

| machine | model | weights | drafts |
|---|---|---|---|
| 16 GB | Gemma 4 26B-A4B | unsloth's UD-IQ2_M GGUF | none at this size |
| 32 GB | Qwen3.6-35B-A3B | ggml-org's Q4_K_M GGUF | its MTP head |
| 32 GB | Gemma 4 26B-A4B | unsloth's UD-Q5_K_M GGUF | Gemma's draft model |
| 64 GB | Qwen3.6-35B-A3B | ggml-org's Q8_0 GGUF | its MTP head |
| 64 GB | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | antirez's Q2 GGUF | its MTP head |
| 128 GB | Qwen3.8-Flash-Next | lily's own 4-bit conversions (q4-xl, q4) | its MTP head |
| 128 GB | GLM 5.3 Flash | antirez's Q2 GGUF | its MTP block |
| 128 GB | DeepSeek V4 Flash | antirez's Q2 GGUF | DSpark |

GGUF files are loaded as they are published, without re-quantization. On the
smaller machines `--memory-limit-gb` keeps everything lily allocates on the
GPU (weights, scratch and the active session) within the budget the machine
leaves the GPU, and fits the longest context that budget allows.

It is developed on a 16-inch M5 Max MacBook Pro with 128 GB (J716c) running
the [aurora-silicon](https://github.com/aurora-silicon) Linux kernel and
Mesa's Honeykrisp Vulkan driver. The server is OpenAI-compatible
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, tools,
`reasoning_content`, streaming) and is used daily as a coding-agent backend.

## Where it comes from

- **[Perplexity's lily](https://github.com/perplexityai/pplx-garden/tree/main/lily)**
  ([write-up](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)):
  the Metal kernel foundations (4-bit weight streaming adapted from MLX,
  tensor-op GEMMs, the MoE, Gated DeltaNet and attention kernels), the
  runtime shader compiler and the server.
- **[fabiogreter/lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next)**,
  the fork this one is forked from: the Qwen3.8-Flash-Next graph and
  converter, the Metal 4 transport, speculative decoding with the model's own
  draft head, the session cache, and the decode and prefill kernel tuning
  the Vulkan kernels follow. The Qwen model and its checkpoints below are
  that work.
- **[antirez/ds4](https://github.com/antirez/ds4)** (MIT): the reference
  implementation that GLM 5.3 Flash and DeepSeek V4 Flash follow, and the
  quantized GGUFs lily loads. lily's results are checked against ds4's official
  continuation fixtures.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT): the graphs
  Qwen3.6 and Gemma 4 follow (and Gemma's draft loop), and the reference
  their scores are checked against on the same files.
- **ggml**: the GGUF block formats (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K,
  IQ2_XXS, IQ2_S, IQ4_NL, MXFP4) the decoders reproduce bit for bit.
- **This fork**: the Vulkan backend and the GLSL port of every kernel, the
  GGUF models above, the memory fits for smaller machines, and the GPU work
  on Linux it needed (below).

`NOTICE` lists the upstream copyrights and licenses.

## Speed on Linux (M5 Max, 128 GB)

All numbers are tokens per second on the J716c under Linux, measured on
2026-11-14 with the code on `main`. Speed on a base M5 or M5 Pro, the usual
chips of the 16 GB and 32 GB machines, has not been measured; their GPUs and
memory bandwidth are smaller, so expect lower numbers there.

### Served: the standard prompt set

The server as each model runs it (the commands below, drafting on where the
configuration has it), five chat requests, greedy, 400 generated tokens
each, `reasoning_effort: low`: four short coding-agent style questions (an
explanation, a Rust function with tests, debugging advice, a shell
one-liner; 40 to 85 prompt tokens) and one long-context request (a
14 000-character GLSL file to summarize; 4 200 to 4 900 prompt tokens,
depending on the tokenizer). Decode is the rate of the 400 tokens; prefill
the server's own measure of the prompt.

| machine | model | decode, 5 prompts | drafts accepted | prefill, long prompt | prefill, short prompts |
|---|---|---:|---:|---:|---:|
| 16 GB | Gemma 4 26B-A4B UD-IQ2_M | 91-100 | (no drafts) | 1 310 | 630-720 |
| 32 GB | Qwen3.6-35B-A3B Q4_K_M | 136-140 | 81-91% | 3 180 | 600-700 |
| 32 GB | Gemma 4 26B-A4B UD-Q5_K_M | 123-138 | 66-86% | 1 110 | 380-470 |
| 64 GB | Qwen3.6-35B-A3B Q8_0 | 127-138 | 78-89% | 3 040 | 570-685 |
| 64 GB | Qwen3.8-Flash-Next Q2 | 76-89 | 68-88% | 1 460 | 250-300 |
| 128 GB | Qwen3.8-Flash-Next q4-xl | 97-118 | 67-89% | 1 680 | 190-440 |
| 128 GB | DeepSeek V4 Flash Q2 | 49-52 | 73-81% | 450 | 130-165 |
| 128 GB | GLM 5.3 Flash Q2 | 36-45 | 56-84% | 510 | 75-110 |

Short prompts spend most of their prefill time on fixed per-request costs,
so their rates say little about the kernels. In daily agent use decode is
often faster than this set, because agent output (tool calls, code copied
from the context) drafts well.

### Each model's bench tool

Plain decoding (no drafts) and prompt processing, measured with each model's
own tool (`lily-bench`, `lily-qwen36`, `lily-qwen38`, `lily-glm53`,
`lily-ds4`, `lily-gemma4`), greedy, on real text:

| model | prefill 1K / 4K | decode, no drafts | decode with drafts |
|---|---:|---:|---:|
| Gemma 4 UD-IQ2_M | 1 820 / 1 500 | ~92 | (121-124 with 2 drafts; off at the 16 GB setting) |
| Gemma 4 UD-Q5_K_M | 1 340 / 1 170 | ~90 | 131-150 (2 drafts) |
| Qwen3.6-35B-A3B Q4_K_M | 3 300-3 600 / ~3 480 | ~105 | 131-157 (served) |
| Qwen3.6-35B-A3B Q8_0 | 5-10% below Q4 | 5-10% below Q4 | |
| Qwen3.8-Flash-Next Q2 | ~1 610 / ~1 900 | ~54 | 61-91 |
| Qwen3.8-Flash-Next q4-xl | ~1 670 / ~2 020 | ~69 | ~96 (2 drafts) |
| DeepSeek V4 Flash Q2 | ~560 / 470-505 | ~36 | 37-57 (DSpark; Italian to code) |
| GLM 5.3 Flash Q2 | ~650 / ~643 | ~37 | 39-49 (Italian to code) |

GLM reads about 10 GB of weights per token, so its plain decode is bounded
near 48 tok/s by memory bandwidth; Qwen3.6 reads about 2 GB, which is why it
is the fastest decoder here.

### Quality

| model | check | lily | reference |
|---|---|---|---|
| Qwen3.8-Flash-Next Q2 | ds4's 100 official continuations, NLL / first tokens | 0.354, 68/100 | ds4 0.353, 70/100 |
| GLM 5.3 Flash Q2 | ds4's 100 official continuations | 0.466, 89/100 | ds4 0.458-0.462, 89-90/100 |
| DeepSeek V4 Flash Q2 | ds4's 100 official continuations | 0.395, 57/100 | ds4 0.398, 56/100 |
| Qwen3.6-35B-A3B Q4_K_M | 24 long texts, NLL per token | 1.837, same top token as llama.cpp 97% | llama.cpp 1.839 on the same file |
| Gemma 4 UD-Q5_K_M | KL divergence from llama.cpp on the same file (chat / human text) | 0.0018 / 0.0106 | |
| Gemma 4 UD-IQ2_M | KL divergence from llama.cpp on the same file | 0.0077 / 0.0263 | |

### Memory

| machine | configuration | everything on the GPU | longest context |
|---|---|---:|---:|
| 16 GB | Gemma 4 UD-IQ2_M, `--memory-limit-gb 10.2` | 11.0 GB | 14 592 |
| 32 GB | Qwen3.6 Q4_K_M, `--memory-limit-gb 22` | 23.6 GB | 52 088 |
| 32 GB | Gemma 4 UD-Q5_K_M, `--memory-limit-gb 22` | 23.6 GB | 41 216 |
| 64 GB | Qwen3.6 Q8_0, `--memory-limit-gb 44` | 47.2 GB | 262 144 |
| 64 GB | Qwen3.8-Flash-Next Q2, `--memory-limit-gb 44.7` | 48.0 GB | 32 760 |
| 128 GB | Qwen3.8-Flash-Next q4-xl | 75 GB allocated, 32 GB n-gram table paged | 131 072 |
| 128 GB | GLM 5.3 Flash Q2 | 100 GB allocated | 65 536 |
| 128 GB | DeepSeek V4 Flash Q2 | 95 GB allocated | 65 536 |

`--memory-limit-gb` counts GiB. A memory-limited configuration serves one
request at a time.

## Running it

Requirements:

- An Apple M5 Max under Linux with the aurora-silicon kernel
  (`integration/aurora`), plus the GPU compute-time allowance
  (aurora-silicon/linux #1752) and, for full decode speed, the GPU mailbox
  interrupt fix (#1844).
- Mesa's Honeykrisp from `aurora-silicon/mesa` (`aurora-wip`) with the M5
  matrix units enabled (`ASAHI_M5_COOPERATIVE_MATRIX=1`). For GLM in
  particular the compute kick (aurora-silicon/mesa #260) matters: without it
  a long prefill pass holds the desktop's GPU work back past the kernel's
  timeout. The barrier and compiler changes in #266, #308 and #309 add a few
  percent.
- The Rust toolchain pinned by `rust-toolchain.toml`.

The weights are mapped read-only from the page cache rather than copied, and
the GPU heap needs more than half of RAM:

```sh
export VK_ICD_FILENAMES=$MESA/share/vulkan/icd.d/asahi_icd.aarch64.json
export LD_LIBRARY_PATH=$MESA/lib
export ASAHI_M5_COOPERATIVE_MATRIX=1 heap_memory_percent=0.84
export hk_host_memory_read_only=true hk_host_memory_import_read_only=true

cargo build --release --locked

# Qwen3.8-Flash-Next
./target/release/lily --model ~/models/Qwen3.8-Flash-Next-lily-q4-xl \
  --bind 127.0.0.1:8000 --max-seq 131072

# Qwen3.6-35B-A3B: 32 GB machines (Q4_K_M), 64 GB machines (Q8_0 with
# --memory-limit-gb 44); the MTP file (mtp-*.gguf) is found beside the model
./target/release/lily --model ~/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --bind 127.0.0.1:8000 --memory-limit-gb 22

# Qwen3.8-Flash-Next from ds4's Q2 GGUF for a 64 GB machine: 44.8 GB of
# weights, and everything lily allocates (weights, scratch, one 32K session)
# within 48.0 GB (44.7 GiB); one request at a time. The n-gram table is paged
# from a lily checkpoint of the model (docs/qwen38-q2.md)
./target/release/lily --model ~/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --ngram-checkpoint ~/models/Qwen3.8-Flash-Next-lily-q4-xl \
  --bind 127.0.0.1:8000 --memory-limit-gb 44.7

# GLM 5.3 Flash (a .gguf path selects the model from its metadata)
./target/release/lily --model ~/models/GLM-5.3-Flash-Q2/GLM-5.3-Flash-Q2.gguf \
  --bind 127.0.0.1:8001 --max-seq 65536 --cache-bytes 9G --pin-weights off \
  --mtp-drafts 1 --max-batch 1

# DeepSeek V4 Flash (DSpark's support file is found beside the model)
./target/release/lily --model ~/models/DeepSeek-V4-Flash-Q2/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --bind 127.0.0.1:8002 --max-seq 65536 --cache-bytes 9G --pin-weights off \
  --mtp-drafts 5 --max-batch 1

# Gemma 4 26B-A4B, 32 GB machines (16 GB: UD-IQ2_M, --memory-limit-gb 10.2
# --mtp-drafts 0 --checkpoints-per-session 1; docs/gemma4.md); the draft
# model is found beside the model
./target/release/lily --model ~/models/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf \
  --bind 127.0.0.1:8003 --memory-limit-gb 22 --mtp-drafts 2 \
  --decode-checkpoint-tokens 0 --pin-weights off
```

Only one model fits in 128 GB at a time; every loader takes a machine-wide
instance lock before mapping weights. [docs/qwen38-q2.md](docs/qwen38-q2.md)
covers the Q2 GGUF of Qwen. [docs/glm53.md](docs/glm53.md) has
GLM's checkpoint, memory, differences from ds4 and what is not implemented
yet; [docs/deepseek4.md](docs/deepseek4.md) the same for DeepSeek V4 Flash
and [docs/gemma4.md](docs/gemma4.md) for Gemma 4, with its files and memory
for 16 GB and 32 GB machines. [src/kernels/vulkan/PORTING.md](src/kernels/vulkan/PORTING.md) describes
the Vulkan port.

## The Linux port

On Linux, `src/vulkan.rs` is `crate::metal`: it implements the Metal
backend's API over Vulkan, so the engine and the Rust dispatch code run
unchanged, and every Metal kernel has a GLSL twin in `src/kernels/vulkan/`
doing the same arithmetic in the same order. Beyond the straight port:

- **Matrix units.** Prefill GEMMs and attention run on the M5's matrix units
  through Vulkan's cooperative-matrix extensions.
- **Kernels shaped for AGX3.** On this GPU a branch waits for every load in
  flight, registers per thread bound how many loads can be kept in flight,
  and dispatches in the same dependency level do not overlap: hot kernels
  are specialized for the model's fixed shapes so their loops are straight
  code, and same-level work is fused into single dispatches.
- **Weights from the page cache.** Checkpoints are imported read-only
  (`VK_EXT_external_memory_host`), not copied into GPU memory.
- **The driver and kernel underneath.** Bringing lily up found and fixed
  problems below it, filed as pull requests to aurora-silicon's Mesa and
  Linux trees: a stall where one client's long compute work starves other
  clients (the Mesa compute kick and a kernel waiting rule), cheaper
  barriers between dependent dispatches, device-scope barrier and bfloat16
  rounding fixes in the AGX3 compiler, flattening of small uniform branches,
  and a GPU mailbox that had fallen back to polling (a 5% decode loss).

## GLM 5.3 Flash

GLM 5.3 Flash (45 layers of KDA linear attention and DSA sparse attention at
3:1, 288 routed experts with one shared, four hyper-connection streams, a
built-in MTP draft block) runs from antirez's Q2 GGUF as published:
`src/gguf.rs` reads the file (hardened against malformed headers), a schema
checks every tensor before anything is mapped, and `src/glm53/` holds the
model. Correctness is checked layer by layer against a CPU reference on the
real weights and end to end against ds4's official continuations.
Speculative decoding uses the MTP block, one draft a step, proposed only when
the draft is confident enough to pay for the two-row verify pass.

## DeepSeek V4 Flash

DeepSeek V4 Flash (43 layers of multi-query attention over a 128-row window
plus compressed rows, ratio 4 with a sparse indexer or ratio 128 by layer,
FP8 KV rounding, 256 routed experts with 6 active and a shared one, four
hyper-connection streams) runs from antirez's Q2 GGUF as published, checked
against a schema of every tensor; `src/deepseek4/` holds the model and a CPU
reference of every operation. Speculative decoding uses DSpark, antirez's
three-stage block drafter (`--mtp-drafts 5`), which drafts up to five tokens
a step and backs off when its drafts keep being rejected.

## Qwen3.6-35B-A3B

Qwen3.6-35B-A3B (40 layers: Gated DeltaNet and gated attention at 3:1, 256
routed experts with 8 active and a gated shared expert, an MTP head) runs
from ggml-org's Q4_K_M and Q8_0 GGUFs, which use only block types lily
already decodes. `src/qwen36/` follows llama.cpp's graph; the attention
kernels are Qwen3.8's, the expert and projection kernels GLM's. Speculative
decoding uses the model's MTP head (`--mtp-drafts 2`, the default), loaded
from ggml-org's separate MTP file.

## Gemma 4 26B-A4B

Gemma 4 26B-A4B (30 layers: 25 sliding-window and 5 global attention
layers, a dense GeGLU MLP and 128 routed experts with 8 active in each)
runs from the published GGUF files, chosen so that everything lily
allocates on the GPU stays under 11 GB on a 16 GB machine (UD-IQ2_M, 14.5K
context) and under 24 GB on a 32 GB one (UD-Q5_K_M, 41K), through
`--memory-limit-gb`. `src/gemma4/`
follows llama.cpp's graph, which is also the reference its scores are
checked against on the same files. Speculative decoding uses Gemma's own
draft model, which reads the target's caches (`--mtp-drafts 2`).

## Qwen3.8-Flash-Next

Qwen's `qwen4_exp` preview architecture: 48 layers (36 Gated DeltaNet, 12
sparse attention with an indexer), a 512-expert MoE with 10 active, a
four-stream gated residual, a 32 GB hashed n-gram embedding, a draft head and
a vision tower. Two conversions, both mostly 4-bit:
**[q4-xl](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl)**
(recommended, 106.9 GB; attention, the shared expert, the LM head and the
embedding at 8 bits) and
**[q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4)**
(105.5 GB). [docs/qwen38-flash-next-checkpoint-format.md](docs/qwen38-flash-next-checkpoint-format.md)
documents the layout; `tools/convert/convert_qwen38_flash_next.py` makes them
from the BF16 weights ([tools/README.md](tools/README.md)).

## Server features

- **Caching.** Each conversation's state stays resident within a GPU memory
  budget, sessions evicted from it move to disk and survive restarts, and
  shared agent preambles are stored once as durable prefixes
  ([docs/architecture.md](docs/architecture.md), "The session cache").
- **Batching.** Up to `--max-batch` requests decode together (Qwen3.8
  q4-xl); the GGUF models serve one request at a time.
- **Thinking controls.** A thinking budget with nudges, and a tool call
  drawn inside the reasoning ends it ([docs/server.md](docs/server.md)).
- **Timings.** Every response carries prefill and decode rates, cached tokens
  and draft acceptance; `tools/opencode-plugin-timings/` shows them in
  opencode.

## Tests

```sh
cargo test --locked
```

runs the kernel tests against CPU references, the shader compilation test and
the API tests on the local GPU through Vulkan; they need no
checkpoint. Tests that need weights are ignored by default and take
`LILY_MODEL_DIR_FLASH` (Qwen3.8), `LILY_QWEN36_GGUF` (Qwen3.6),
`LILY_GLM53_GGUF` (GLM), `LILY_DS4_GGUF` (DeepSeek) or `LILY_GEMMA4_GGUF`
(Gemma).
[CONTRIBUTING.md](CONTRIBUTING.md) has the gates.

## License

Apache-2.0. `NOTICE` records the upstream copyrights and the MLX, MLX-LM,
ds4 and llama.cpp/ggml (MIT) code adapted here. The models are under their own licences, stated
on their checkpoints' cards.
