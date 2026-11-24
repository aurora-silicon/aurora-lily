# lily on Linux

Fast local inference of mixture-of-experts and hybrid language models on
Apple Silicon running Linux. lily is a compact inference server with
hand-written GPU kernels; this fork runs it on Linux through Vulkan (Mesa's
Borealis driver) and serves eight model families, from a 16 GB laptop to a
128 GB workstation. The server is OpenAI-compatible (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, tools, `reasoning_content`, streaming) and
is used daily as a coding-agent backend.

| Memory | Model              | Weights                              | Speculative drafts            |
|--------|--------------------|--------------------------------------|-------------------------------|
| 16 GB  | Qwen3.5-4B / 9B    | Q4_K conversion below                | embedded MTP head             |
| 16 GB  | Gemma 4 26B-A4B    | unsloth's UD-IQ2_M GGUF              | optional: Gemma's draft model |
| 16 GB  | Inkling 2 Mini     | unsloth's UD-Q2_K_XL / UD-IQ2_XXS    | optional: DSpark (Q8_0)       |
| 32 GB  | Qwen3.6-35B-A3B    | ggml-org's Q4_K_M GGUF               | its MTP head                  |
| 32 GB  | Gemma 4 26B-A4B    | unsloth's UD-Q5_K_M GGUF             | Gemma's draft model           |
| 32 GB  | Inkling 2 Mini     | ggml-org's IQ4_NL / unsloth UD-Q4_K_XL | DSpark (Q8_0)               |
| 32 GB  | Agnes 3.0 Flash    | lily's q4_k conversion               | its MTP head                  |
| 64 GB  | Qwen3.6-35B-A3B    | ggml-org's Q8_0 GGUF                 | its MTP head                  |
| 64 GB  | Inkling 2 Mini     | ggml-org's Q8_0 GGUF                 | DSpark (Q8_0 / BF16)          |
| 64 GB  | Qwen3.8-Flash-Next | antirez's Q2 GGUF                    | its MTP head                  |
| 64 GB  | Agnes 3.0 Flash    | lily's q8_0 conversion               | its MTP head                  |
| 128 GB | Qwen3.8-Flash-Next | lily's 4-bit conversions (q4-xl, q4) | its MTP head                  |
| 128 GB | Inkling 2 Mini     | ggml-org's BF16 GGUF                 | DSpark (BF16)                 |
| 128 GB | GLM 5.3 Flash      | antirez's Q2 GGUF                    | its MTP block                 |
| 128 GB | DeepSeek V4 Flash  | antirez's Q2 GGUF                    | DSpark                        |

For rows marked as conversions, use the model's quantization recipe.
The other GGUFs load as published. On the smaller
machines `--memory-limit-gb` keeps everything lily allocates on the GPU
(weights, scratch and the active session) within the budget the machine
leaves the GPU, and fits the longest context that budget allows.

## Hardware

- **M5 family** (GPUs with matrix units): every model above. Developed and
  measured on a 16-inch M5 Max MacBook Pro with 128 GB. The base M5 and M5
  Pro use the same kernels; their GPUs and memory bandwidth are smaller, so
  expect lower numbers than the tables below.
- **M4 family** (no matrix units): Qwen3.5-4B and 9B run with embedded MTP
  heads. Gemma 4 and Inkling 2 Mini run with tiled GEMMs and decode kernels
  tuned for the M4's GPU (Inkling 2 Mini also runs its DSpark speculative
  drafter on the GPU, [docs/inkling2.md](docs/inkling2.md)). Agnes 3.0 Flash
  has a path without matrix units too, verified on another GPU without them
  but not yet measured on an M4 ([docs/agnes3.md](docs/agnes3.md)). The
  other models' kernels need matrix units and are refused on these GPUs.
  Mesa offers the M4 8 x 8 cooperative matrices, which run on the same ALUs
  as its FMAs; lily's matrix kernels need the M5's 16 x 16 ones, so the M4
  keeps the tiled GEMM path.
- **M1 to M3 families** (no matrix units either): these take the same path
  as the M4, selected the same way, so the same models are offered and the others
  are refused. Measured once on a 16-inch M1 Max (64 GB, Linux, Mesa
  Borealis) with Gemma 4 UD-IQ2_M: about 42 tokens/s decode and about
  300 tokens/s prefill on a 4 452-token prompt, loaded in 4.9 s
  ([#38](https://github.com/aurora-silicon/aurora-lily/issues/38)). Other
  models and flags on these GPUs are untested.

## Speed

### Linux, M5 Max (128 GB)

The server as each configuration runs it (the commands below), on five chat requests, greedy, 400
generated tokens each: four short coding-agent questions (40–90 prompt tokens) and one long-context
request (a 14 000-character source file to summarize, 4 200–4 900 prompt tokens). Tokens per second,
measured 2026-11-15 on `main` (7838f9b) with Mesa `aurora-wip` plus the AGX compiler pull requests
listed under [Running it](#running-it):

| Memory | Model              | Weights   |  Decode | Drafts accepted | Prefill, long prompt | Prefill, short prompts |
|--------|--------------------|-----------|--------:|----------------:|---------------------:|-----------------------:|
| 16 GB  | Gemma 4            | UD-IQ2_M  | 123–131 |               – |                3 400 |                680–940 |
| 32 GB  | Qwen3.6            | Q4_K_M    | 138–149 |          82–87% |                3 710 |                580–860 |
| 32 GB  | Gemma 4            | UD-Q5_K_M | 150–169 |          67–83% |                3 460 |                570–780 |
| 64 GB  | Qwen3.6            | Q8_0      | 138–145 |          83–90% |          3 170–3 450 |                570–690 |
| 64 GB  | Qwen3.8-Flash-Next | Q2        |   80–97 |          67–88% |                1 640 |                260–300 |
| 128 GB | Qwen3.8-Flash-Next | q4-xl     |  93–119 |          67–89% |                1 830 |                230–410 |
| 128 GB | DeepSeek V4 Flash  | Q2        |   51–58 |          67–89% |                  770 |                130–170 |
| 128 GB | GLM 5.3 Flash      | Q2        |   41–46 |          66–83% |                  520 |                100–190 |

- Decode is the range over the five requests; drafts are on wherever the configuration has a draft
  model or head (the 16 GB Gemma setting leaves them off to save memory).
- Short prompts mostly measure fixed per-request costs. Agent output (tool calls, code copied from the
  context) drafts well, so decode in agent use is often faster than this set.
- On the 140 W adapter, minutes of continuous GPU load can exceed what the adapter and battery supply,
  and the GPU then clocks down. These are short runs; long batch jobs may see lower numbers.

### Compared with macOS

The same M5 Max under macOS, as published by the projects lily on Linux follows:

| Model                    | Engine on macOS   | Prefill, macOS     | Prefill, Linux          | Decode, macOS               | Decode, Linux                |
|--------------------------|-------------------|--------------------|-------------------------|-----------------------------|------------------------------|
| Qwen3.8-Flash-Next q4-xl | lily (Metal) [^1] | 2 410–2 524        | 1 830                   | 77–80 plain, 98–107 drafted | ~69 plain, 93–119 drafted    |
| DeepSeek V4 Flash Q2     | ds4 (Metal) [^2]  | 790 (2K), 710 (4K) | ~1 000 (2K), 770 (4.4K) | 39 plain                    | ~40 plain, 51–58 with DSpark |

[^1]: [fabiogreter's lily](https://github.com/fabiogreter/lily-qwen3.8-flash-next), `docs/performance.md`
    (2026-10-12): fresh prompts over HTTP, 4K–64K tokens, 2 drafts.
[^2]: [antirez's ds4](https://github.com/antirez/ds4), `speed-bench/m5_max.csv`.

Decode on Linux matches or beats macOS; Qwen3.8-Flash-Next's prefill is still about 25% behind lily on Metal.
Neither project publishes M5 Max numbers for the other models.

### Linux, M4 (16 GB)

Measured 2026-11-24 on a 16 GB M4 MacBook Air (J713, 10-core CPU, 8-core GPU, fanless, on AC) at
`main` (`5a447c6`) with Linux 7.3.0 and Mesa Borealis, cool start below 54 °C. Rates are tokens per
second from the model `bench`, `spec`, and `profile` commands (64 greedy output tokens):

| Model           | Weights    |     Prefill | Decode, plain | Decode, drafted | Notes                                                        |
|-----------------|------------|------------:|--------------:|----------------:|--------------------------------------------------------------|
| Qwen3.5-4B      | Q4_K       |       203.1 |          30.5 |       35.4–37.0 | 1K / 256p; `--drafts 2`, 83–86% accepted                     |
| Gemma 4 26B-A4B | UD-IQ2_M   | 173.2–177.6 |     27.3–30.4 |       37.9–38.0 | 256p–1K; `--drafts 2`, 73% accepted on a coding prompt       |
| Inkling 2 Mini  | UD-Q2_K_XL |       100.1 |     19.1–19.8 |            30.6 | 256p bench; drafted is `profile` `verify(4)` (3.27 tok/step) |

- **Workload and parity.** Qwen3.5-4B and Gemma 4 `bench` use `src/qwen36/model.rs` (256 and 1 024
  prompt tokens); Inkling 2 Mini uses `--memory-limit-gb 10.6` (256-token prompt) with DSpark
  (`Q8_0`). Speculative decoding matches plain greedy output verbatim (`64/64` tokens on Qwen3.5-4B
  and Gemma 4; `128/128` chat tokens on Inkling 2 Mini).
- **Fanless thermal throttling.** The M4 Air clocks down under sustained GPU load as the die reaches
  80–100 °C. Heat-soaked runs measure lower than cool starts (and multi-request
  served Gemma 4 decode drops to about 19 tok/s once warm).
- **Qwen3.5-9B.** Also fits in 16 GB (6.7 GB weights and scratch versus 3.2 GB for 4B) using the
  Q4_K recipe under [Running it](#running-it), decoding around 11–18 tok/s in short coding runs.
- **Served overhead.** Serving over HTTP (`/v1/chat/completions`) adds fixed per-request framing
  overhead on short prompts; see [docs/gemma4.md](docs/gemma4.md) and
  [docs/inkling2.md](docs/inkling2.md) for per-model notes and memory flags.

### Quality

Against ds4's 100 official continuations, on the same GGUF files ds4 runs:

| Model                 | NLL, lily |    NLL, ds4 | First tokens right, lily | First tokens right, ds4 |
|-----------------------|----------:|------------:|-------------------------:|------------------------:|
| Qwen3.8-Flash-Next Q2 |     0.354 |       0.353 |                   68/100 |                  70/100 |
| DeepSeek V4 Flash Q2  |     0.396 |       0.398 |                   55/100 |                  56/100 |
| GLM 5.3 Flash Q2      |     0.465 | 0.458–0.462 |                   91/100 |               89–90/100 |

Against llama.cpp on the same file:

| Model           | Weights   | Check                                           | lily            |
|-----------------|-----------|-------------------------------------------------|-----------------|
| Qwen3.6-35B-A3B | Q4_K_M    | NLL per token, 24 long texts                    | 1.813 [^3]      |
| Qwen3.6-35B-A3B | Q8_0      | NLL per token, 24 long texts                    | 1.812 [^3]      |
| Gemma 4         | UD-Q5_K_M | KL divergence from llama.cpp, chat / human text | 0.0017 / 0.0107 |
| Gemma 4         | UD-IQ2_M  | KL divergence from llama.cpp, chat / human text | 0.0080 / 0.0282 |

[^3]: Within 0.002 of llama.cpp before the chunked prefill (#24), whose bf16 operands added 0.001
    (Q4_K_M) and 0.003 (Q8_0).

Speculative output equals plain greedy decoding up to rounding ties.

### Memory

| Memory | Configuration                                                | Allocated on the GPU                     | Longest context                      |
|--------|--------------------------------------------------------------|------------------------------------------|--------------------------------------|
| 16 GB  | Gemma 4 UD-IQ2_M, `--memory-limit-gb 10.2`                   | 11.0 GB                                  | 21 248 (10 496 with the draft model) |
| 16 GB  | Inkling 2 Mini UD-Q2_K_XL, `--memory-limit-gb 10.6` (M4 Air) | 11.4 GB                                  | 37 852 (with DSpark, 1 snapshot)     |
| 32 GB  | Qwen3.6 Q4_K_M, `--memory-limit-gb 22`                       | 23.6 GB                                  | 52 088                               |
| 32 GB  | Gemma 4 UD-Q5_K_M, `--memory-limit-gb 22`                    | 23.6 GB                                  | 55 296                               |
| 32 GB  | Agnes 3.0 Flash q4_k, `--memory-limit-gb 22`                 | 23.6 GB                                  | 77 428 (123 688 at 25 on 36 GB)      |
| 64 GB  | Qwen3.6 Q8_0, `--memory-limit-gb 44`                         | 47.2 GB                                  | 262 144                              |
| 64 GB  | Qwen3.8-Flash-Next Q2, `--memory-limit-gb 44.7`              | 48.0 GB                                  | 73 720 (163 832 without drafts)      |
| 128 GB | Qwen3.8-Flash-Next q4-xl                                     | 75 GB, plus the 32 GB n-gram table paged | 131 072                              |
| 128 GB | GLM 5.3 Flash Q2                                             | 100 GB                                   | 65 536                               |
| 128 GB | DeepSeek V4 Flash Q2                                         | 95 GB                                    | 65 536                               |

`--memory-limit-gb` counts GiB (the GPU column is decimal GB). Except for the M4 Air row, fits are
measured on the M5 Max; on the M4 Air, Gemma 4 uses 512-row prompt runs and fits 21 760 tokens
in 10.1 GB. A memory-limited configuration serves one request at a time.

## Running it

You need:

- Linux on Apple Silicon with the [aurora-silicon](https://github.com/aurora-silicon)
  kernel (`main`), including the GPU compute-time allowance
  (aurora-silicon/linux #1752); on the M5 Max, the GPU mailbox interrupt fix
  (#1844) for full decode speed.
- Mesa's Borealis from aurora-silicon/mesa `aurora-wip` (M5 cooperative matrix
  is enabled by default). The AGX compiler
  pull requests #370 to #375, #379 and #380 are in the build measured above;
  they add 2–15% to prefill.
- The Rust toolchain pinned by `rust-toolchain.toml`.

Weights are mapped read-only from the page cache rather than copied, and the
GPU heap needs more than half of RAM:

```sh
export VK_ICD_FILENAMES=$MESA/share/vulkan/icd.d/asahi_icd.aarch64.json
export LD_LIBRARY_PATH=$MESA/lib
export heap_memory_percent=0.84
export hk_host_memory_read_only=true hk_host_memory_import_read_only=true

cargo build --release --locked
```

Then one of (a `.gguf` path selects the model from its metadata; draft
models and support files are found beside the model):

```sh
# 16 GB: Qwen3.5-4B (or 9B), quantized with the recipe below
./target/release/lily --model ~/models/Qwen3.5-4B-Q4_K.gguf \
  --max-seq 16384 --memory-limit-gb 9 --mtp-drafts 2

# 16 GB: Gemma 4, everything under 11 GB (add --mtp-drafts 2 with the draft model)
./target/release/lily --model ~/models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf \
  --memory-limit-gb 10.2 --mtp-drafts 0 --checkpoints-per-session 1 \
  --decode-checkpoint-tokens 0 --pin-weights off

# 16 GB: Inkling 2 Mini UD-Q2_K_XL with DSpark beside the model (docs/inkling2.md)
./target/release/lily --model ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf \
  --memory-limit-gb 10.6 --mtp-drafts 3 --checkpoints-per-session 1 \
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

### Quantizing Qwen3.5 (16 GB)

Start from bartowski's BF16 text GGUF (which includes the embedded MTP layer).
Lily's fused recurrent projections need a common quantization type, so this
recipe uses Q4_K for those projections and Q8_0 for the vocabulary matrices
(use a separate directory for each model, with a recent `llama.cpp` build):

```sh
size=4B  # or 9B
file="Qwen_Qwen3.5-${size}-bf16.gguf"
curl -fL --retry 3 -o "$file" \
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-${size}-GGUF/resolve/main/$file"
llama-quantize --pure --token-embedding-type q8_0 --output-tensor-type q8_0 \
  --tensor-type ssm_alpha=q4_k --tensor-type ssm_beta=q4_k \
  "$file" "Qwen3.5-${size}-Q4_K.gguf" Q4_K
```

## The models

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**: Qwen's `qwen4_exp` preview: 48 layers (36 Gated
  DeltaNet, 12 sparse attention with an indexer), 512 experts with 10
  active, a four-stream gated residual, a 32 GB hashed n-gram embedding and
  a draft head. Two 4-bit conversions,
  **[q4-xl](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl)**
  (recommended, 106.9 GB; attention, shared expert, LM head and embedding at
  8 bits) and **[q4](https://huggingface.co/fabiogreter/Qwen3.8-Flash-Next-lily-q4)**
  (105.5 GB), made by `tools/convert/convert_qwen38_flash_next.py`
  ([format](docs/qwen38-flash-next-checkpoint-format.md)), and antirez's Q2
  GGUF for 64 GB machines ([docs/qwen38-q2.md](docs/qwen38-q2.md)).
- **Qwen3.5-4B / 9B**: dense Gated DeltaNet and gated attention models with
  embedded MTP heads, served through the Qwen3.6 code for 16 GB machines.
- **Qwen3.6-35B-A3B**: Gated DeltaNet and gated attention at 3:1, 256
  experts with 8 active and a gated shared expert, an MTP head loaded from
  ggml-org's separate MTP file.
- **[Agnes 3.0 Flash](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash)**: the
  open-weight Preview, Qwen3.5's dense decoder (Gated DeltaNet and gated
  attention at 3:1, a SwiGLU MLP, an MTP layer) with 64 trained layers of its
  72; lily serves `tools/convert/convert_agnes3_flash.py`'s GGUFs through the
  Qwen3.6 code ([docs/agnes3.md](docs/agnes3.md)).
- **Gemma 4 26B-A4B**: 25 sliding-window and 5 global attention layers, a
  dense GeGLU MLP and 128 experts with 8 active in every layer; drafts from
  Gemma's own draft model, which reads the target's caches
  ([docs/gemma4.md](docs/gemma4.md)).
- **Inkling 2 Mini**: 26.4B total / ~3.4B–5.4B active looped MoE (24 physical
  layers with layers 6–17 looped 3× for 48 logical visits, 128 routed experts
  with 8 active and SITU activation, relative-position attention with zero
  sink, and causal short convolutions); optional GPU DSpark speculative
  drafter ([docs/inkling2.md](docs/inkling2.md)).
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
  without them (the M4) the same work runs as tiled GEMMs that decode the
  quantized weights into `f16` threadgroup memory, accumulate FP16 partial
  sums per K-step, and widen into `f32` accumulators (falling back to `f32`
  on FP16 overflow).
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
- **Batching.** Up to `--max-batch` requests decode together on Qwen3.8-Flash-Next
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
weights are ignored by default and take `LILY_MODEL_DIR_FLASH` (Qwen3.8-Flash-Next),
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
  kernels follow. The Qwen3.8-Flash-Next checkpoints are this work.
- **[antirez/ds4](https://github.com/antirez/ds4)** (MIT): the reference
  implementation GLM 5.3 Flash and DeepSeek V4 Flash follow, the quantized
  GGUFs lily loads, and the official continuation fixtures lily is scored on.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT): the graphs
  Qwen3.6, Agnes 3.0 Flash, Gemma 4 and Inkling 2 Mini follow (plus Gemma's
  draft loop and Inkling 2's DSpark drafter), and the reference Qwen3.6 and
  Gemma 4 are scored against; **ggml** for the GGUF block formats (Q8_0, Q4_0,
  Q4_K, Q5_K, Q6_K, Q2_K, Q2_0, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ4_NL, IQ4_XS, MXFP4, NVFP4), which lily's decoders reproduce bit for
  bit, and the reference quantizers lily's converter ports.

## License

Apache-2.0. `NOTICE` records the upstream copyrights and the MLX, MLX-LM,
ds4 and llama.cpp/ggml (MIT) code adapted here. The models are under their
own licences, stated on their checkpoints' cards.
