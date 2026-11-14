# DeepSeek V4 Flash

lily serves DeepSeek V4 Flash from antirez's published Q2 GGUF loaded as it
is, on the Linux/Vulkan backend, with speculative decoding from the DSpark
support file published beside it. The implementation follows antirez's ds4
(MIT, credited in `NOTICE`).

## Checkpoint and backend

- File: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
  86.7 GB (routed experts IQ2_XXS gate/up and Q2_K down, attention
  projections, shared experts and the output head Q8_0, the rest F16 and
  F32). `src/deepseek4/schema.rs` checks every one of its 1 328 tensors
  (shape and block type) before anything is mapped, and the loader refuses
  anything else.
- DSpark: `DeepSeek-V4-Flash-DSpark-support-0731.gguf` (6.0 GB), found
  beside the model (any `*DSpark*.gguf` in its directory) or named by
  `LILY_DS4_DSPARK`.
- Backend: Vulkan only (GLSL kernels in `src/kernels/vulkan/deepseek4.comp`,
  plus GLM 5.3's GEMV, GEMM, MoE and hyper-connection kernels), written and
  measured on an M5 Max under Honeykrisp.
- The weights are mapped read-only from the page cache, not copied. Run with
  `hk_host_memory_read_only=true hk_host_memory_import_read_only=true`.

## Running it

```sh
lily --model ~/models/DeepSeek-V4-Flash-Q2/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
    --max-seq 65536 --cache-bytes 9G --pin-weights off --max-batch 1 [--mtp-drafts 5]
```

A `.gguf` path as `--model` selects the model from the file's architecture
(`deepseek4`); its metadata stands in for `config.json`, its tokenizer and
chat template are read from it.

- `--mtp-drafts 5`: speculative decoding with DSpark, up to five drafts a
  step (DSpark's block size). Without the flag the support file is not
  loaded.
- `--cache-bytes 9G`, `--pin-weights off`: as for GLM 5.3
  ([glm53.md](glm53.md)). A session is small: a 128-row raw KV window per
  layer (5.6 MB) plus about 6.9 KB a token of compressed rows.
- Tool calls use DeepSeek's DSML format (`<｜DSML｜tool_calls>` blocks of
  `invoke` and `parameter` elements); the server parses them into OpenAI
  tool calls, several per block, streamed. Thinking follows the chat
  template's `<think>` / `</think>`, and the thinking budget works as for the
  other models ([server.md](server.md)).
- One model on the machine at a time: the loader takes the instance lock
  before mapping anything. The page cache has to hold the 92.7 GB of both
  files, so neither GLM 5.3 nor Qwen3.8 fits beside it.

As with GLM, run it on a Mesa with the compute kick
(aurora-silicon/mesa #260): a prompt pass of 2 048 rows is one long compute
command, and without the kick the desktop's GPU work waits behind it.

Memory: the weights in the page cache, about 1 GB of prompt scratch (runs of
2 048 rows), DSpark's scratch and support rings, and the sessions.

## What runs

`src/deepseek4/` holds the configuration (checked against ds4's shape for
the Flash model), the schema check, the model, the DSpark drafter, the engine
interface, the tokenizer and template built from the GGUF, and a CPU
reference of every operation (`reference.rs`, after ds4.c's CPU path).

The trunk: 43 layers of multi-query attention (64 heads of 512, one KV head,
RoPE on the last 64 dimensions) over a 128-row raw window plus compressed
rows, with attention sinks. Layers 2 to 42 alternate between ratio-4
compression with an indexer that selects the top 512 compressed rows (64
heads of 128, Hadamard and FP4 rounding of its rows and queries) and ratio
128 compression; compressed layers use YaRN. KV rows go through ds4's FP8
(E4M3) round trip; caches are f16. The output projection is grouped (8
groups of rank 1 024). The MoE has 256 routed experts, 6 a token, scored by
sqrt-softplus plus a bias (layers 0 to 2 route by a token hash table), and a
shared expert. Four hyper-connection streams with the Sinkhorn split, and a
sigmoid-weighted collapse before the output head.

Decode and verify rows use the GEMV kernels on f32 activations (up to 8 rows
a pass); prompts run in runs of 2 048 rows on the matrix units, one pass a
layer, with routed experts grouped by expert.

DSpark: three stages over the mean hyper-connection state after the trunk's
last three layers, drafting a block of five positions at once (non-causal
attention over the block and a 128-row window of the target's features), a
Markov head on the previous token and a confidence head that ends the block
below 0.6. The verify pass does not write the trunk's or DSpark's rings;
`finish_speculation` commits only the accepted rows. A scheduler after ds4's
skips drafting for a few steps when drafts keep being rejected, so text that
drafts badly costs little against plain decoding. Greedy speculative output
matched plain decoding's tokens in the runs below, but as with GLM, one-row
and multi-row evaluations can round differently, so token identity is not
guaranteed.

## Differences from ds4

- Prompt-run GEMMs round both operands to bfloat16 (f32 accumulators); the
  hyper-connection control projection stays in f32 (in bfloat16 it cost 6 of
  100 first tokens on ds4's fixture). Decode works on f32 activations.
- E4M3 and E2M1 rounding is computed in closed form instead of ds4's table
  search; the unit tests check the two agree on every input class.
- RoPE tables are computed on the host with the CPU reference's arithmetic
  and read by the kernels.
- DSpark seeds the last fed position's features into the support ring
  instead of carrying it as block row 0 (the same rows reach attention).

## Tests and tools

```sh
cargo test --release --lib -- deepseek4     # kernel tests need a GPU, no file
LILY_DS4_GGUF=<path> cargo test --release --lib -- --ignored deepseek4
```

The CPU reference's QAT, RoPE and compressor ops are checked against goldens
computed by ds4's own C functions (`tools/reference/deepseek4/ops_golden.c`).
`lily-ds4` (`src/bin/lily-ds4.rs`) compares the GPU with the CPU reference
layer by layer (`compare`), scores ds4's official DeepSeek V4 Flash
continuations from its `gguf-tools/quality-testing` (`score`), generates
greedy text (`generate`) and measures prefill, decode and DSpark (`bench`,
`spec`).

Measured on the M5 Max, 2026-11-14: ds4's 100 official cases NLL 0.395 per
token and 57/100 first tokens (ds4's own reference for this file 0.398 and
56/100). Prefill about 560 tok/s at 1K tokens and 470 to 505 at 4K; decode
36 tok/s at 1K, 34 at 4K. Greedy DSpark, 256 tokens: code 57 to 58 tok/s
(76% of drafts kept, 2.9 tokens a step) against 38 plain; English prose 41
against 38; raw Italian text 37, the same as plain.

## Not implemented

Batching across sessions (the server decodes one request at a time), sessions
saved to disk, vision.
