# lily on Linux: Qwen3.8-Flash-Next, GLM 5.3 Flash, DeepSeek V4 Flash and Gemma 4 on the Apple M5 Max

This fork runs [lily](https://github.com/perplexityai/pplx-garden/tree/main/lily),
a compact inference server with hand-written GPU kernels, on **Linux on an
Apple M5 Max**, through Vulkan, and serves more than one model family:

| model | weights | status |
|---|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | lily's own 4-bit conversions (q4-xl, q4) | runs |
| Qwen3.8-Flash-Next (64 GB tier) | antirez's Q2 GGUF, loaded as it is | runs |
| GLM 5.3 Flash | antirez's Q2 GGUF, loaded as it is | runs |
| DeepSeek V4 Flash | antirez's Q2 GGUF and DSpark file, loaded as they are | runs |
| Gemma 4 26B-A4B (16 GB and 32 GB machines) | unsloth's UD-IQ2_M and UD-Q5_K_M GGUFs and Gemma's draft model, loaded as they are | runs |

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
- **ggml**: the GGUF block formats (Q8_0, Q4_K, Q2_K, IQ2_XXS) the decoders
  reproduce bit for bit.
- **This fork**: the Vulkan backend and the GLSL port of every kernel, GLM
  5.3 Flash and DeepSeek V4 Flash support, and the GPU work on Linux it
  needed (below).

`NOTICE` lists the upstream copyrights and licenses.

## Speed on Linux (M5 Max, 128 GB)

All numbers are tokens per second on the J716c under Linux, greedy, measured
with lily's own tools between 2026-11-13 and 2026-11-14.

**Qwen3.8-Flash-Next (q4-xl)** (`lily-bench`, a 1 024-token prompt of real
text unless stated):

| | tok/s |
|---|---:|
| prefill, 1K prompt | ~1 670 |
| prefill, 4K prompt | ~2 020 |
| decode, no drafts | ~69 |
| decode, 2 drafts per step (the served default) | ~96 |
| served agent-style prompts, decode | 96 to 114 |

**GLM 5.3 Flash (Q2 GGUF)** (`lily-glm53 bench` / `spec`):

| | tok/s |
|---|---:|
| prefill, 1K / 4K / 8K prompt | ~650 / ~643 / ~613 |
| decode, no drafts (27.1 ms a token) | ~37 |
| decode with the MTP draft, code | ~49 |
| decode with the MTP draft, English prose | ~43 |
| decode with the MTP draft, raw Italian text | ~39 |
| served agent-style prompts, decode | 37 to 46 |

GLM reads about 10 GB of weights per token (Qwen about 4.4), so plain decode
is bounded near 48 tok/s by memory bandwidth. On ds4's 100 official GLM 5.3
Flash continuations lily scores an NLL of 0.466 per token with 89/100 first
tokens matched, close to ds4's own references (0.458 to 0.462, 89 to 90/100).

**DeepSeek V4 Flash (Q2 GGUF)** (`lily-ds4 bench` / `spec`):

| | tok/s |
|---|---:|
| prefill, 1K / 4K prompt | ~560 / ~470-505 |
| decode, no drafts | ~36 |
| decode with DSpark, code | ~57 |
| decode with DSpark, English prose | ~41 |
| decode with DSpark, raw Italian text | ~37 |

On ds4's 100 official DeepSeek V4 Flash continuations lily scores an NLL of
0.395 per token with 57/100 first tokens matched (ds4's reference for this
file: 0.398, 56/100).

In daily agent use, short turns spend most of their prefill time on fixed
per-request costs, and decode is often faster than these numbers because
agent output (tool calls, code copied from the context) drafts well.

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

# Gemma 4 26B-A4B, 32 GB machines (16 GB: UD-IQ2_M, --max-seq 8192
# --cache-bytes 0.6G, no drafts); the draft model is found beside the model
./target/release/lily --model ~/models/gemma-4-26B-A4B-it-UD-Q5_K_M.gguf \
  --bind 127.0.0.1:8003 --max-seq 32768 --cache-bytes 2G --pin-weights off \
  --mtp-drafts 2
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

## Gemma 4 26B-A4B

Gemma 4 26B-A4B (30 layers: 25 sliding-window and 5 global attention
layers, a dense GeGLU MLP and 128 routed experts with 8 active in each)
runs from the published GGUF files, chosen so that everything lily
allocates on the GPU stays under 11 GB on a 16 GB machine (UD-IQ2_M, 8K
context) and under 24 GB on a 32 GB one (UD-Q5_K_M, 32K). `src/gemma4/`
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
- **Batching.** Up to `--max-batch` requests decode together (Qwen).
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
`LILY_MODEL_DIR_FLASH` (Qwen), `LILY_GLM53_GGUF` (GLM) or `LILY_DS4_GGUF`
(DeepSeek).
[CONTRIBUTING.md](CONTRIBUTING.md) has the gates.

## License

Apache-2.0. `NOTICE` records the upstream copyrights and the MLX, MLX-LM and
ds4 (MIT) code adapted here. The models are under their own licences, stated
on their checkpoints' cards.
