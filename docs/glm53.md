# GLM 5.3 Flash

lily also serves GLM 5.3 Flash, from antirez's published Q2 GGUF loaded as it
is, on the Linux/Vulkan backend. The implementation follows antirez's ds4
(MIT, credited in `NOTICE`), which runs the same file on Metal, CUDA and ROCm.

## Checkpoint and backend

- File: `GLM-5.3-Flash-Q2.gguf` from `antirez/glm-5.3-flash-gguf`, 96.5 GB
  (routed experts IQ2_XXS gate/up and Q2_K down, the rest Q8_0, Q4_K, BF16
  and F32). Other GLM GGUFs load only if every tensor has the shape and one of
  the block types the kernels read, and the tensors one dispatch reads
  together share a type (`src/glm53/schema.rs`); the loader refuses anything
  else before mapping it.
- Backend: Vulkan only (GLSL kernels in `src/kernels/vulkan/glm53*.comp`),
  written and measured on an M5 Max under Honeykrisp. There are no Metal
  kernels yet.
- The weights are mapped read-only from the page cache, not copied. Run with
  `hk_host_memory_read_only=true hk_host_memory_import_read_only=true`.

## Running it

```sh
lily --model ~/models/GLM-5.3-Flash-Q2/GLM-5.3-Flash-Q2.gguf \
    --cache-bytes 9G --pin-weights off [--mtp-drafts 1] [--max-seq 65536]
```

A `.gguf` path as `--model` selects GLM (the file's architecture is the
model type; its metadata stands in for `config.json`, its tokenizer and chat
template are read from it).

- `--cache-bytes 9G`: the session cache's floor for models with bf16 caches is
  8.6 GB; GLM's sessions are small (about 13 KB a token of context plus
  0.3 GB of recurrent state), so this is a cap, not an allocation.
- `--pin-weights off`: pinning would lock all 96.5 GB of weights in memory.
- `--mtp-drafts 1`: speculative decoding with the model's own MTP block, one
  draft a step. A draft is proposed only when its probability under the MTP
  head (softmax at temperature 1 over the whole vocabulary, before the
  request's penalties or truncation) is at least 0.3; otherwise the step is
  an ordinary one-row step. This is an empirical scheduling heuristic for
  greedy decoding on the M5 Max (a two-row verify pass costs about 37 ms
  against 28 for one row, so a draft pays above about 30% acceptance;
  greedy drafts below 0.3 were accepted about a quarter of the time), not a
  calibrated acceptance probability under every sampler.
  It changes which steps verify a draft, not the sampling rule.
- One model on the machine at a time: the GLM loader takes the instance lock
  (`src/instance.rs`) before mapping anything, like every other loader, and
  GLM and Qwen3.8-Flash-Next do not fit in 128 GB together.

On the M5 Max a long compute command holds other GPU clients back until it
ends; a per-layer prefill pass processing 1 024 rows is one, and without
Mesa's compute kick (aurora-silicon/mesa #260) the desktop's work can wait
past the kernel driver's timeout. Run GLM on a Mesa with the kick.

Memory: the 96.5 GB of weights in the page cache, about 1.5 GB of scratch
for prefill runs of 1 024 rows, and the sessions.

## What runs

`src/gguf.rs` reads the file (hardened against malformed headers) and decodes
its blocks on the CPU. `src/glm53/` holds the configuration, the schema check,
the model (45 trunk layers of KDA linear attention and DSA sparse attention,
3:1, three dense FFN layers then 288 routed experts and a shared one, four
hyper-connection streams with the Sinkhorn split), the MTP draft block, the
engine interface, the tokenizer and template built from the GGUF, and a CPU
reference of every operation. Decode and the speculative verify pass use the
GEMV kernels on f32 activations (up to four rows); prompts run in runs of
1 024 rows on the matrix units, routed experts grouped by expert, one pass a
layer. Speculation uses the trunk sampler with the same request steps and
penalty accounting as ordinary decoding. One-row and multi-row
floating-point evaluations can differ, so token identity with ordinary
decoding is not guaranteed: a near-tied top-8 expert boundary, a near-tied
final argmax, a sampled draw near a cumulative-probability boundary or a
sparse-attention pool selection can go the other way. In the investigated
raw C text (`lily-glm53 spec-diff`, which looks for the first routing
difference among the rows both decodings fed with the same tokens), a
near-tied expert boundary is the first observed routing difference
(decode token 20, layer 24: a CPU-reconstructed top-8 score gap of 1.9e-6,
under an FFN-input relative L2 difference of 2.2e-5; at most 4.6e-5 at
every earlier row and layer), followed by state drift and an output
difference at token 119, where plain decoding's top two logits were 0.066
apart. Plain decoding itself meets score gaps below 1e-5 about once every
14 tokens on that text. The same kind of tie was seen to 1e-6 after a
pseudo-random prompt.
Decode steps run synchronously (`forward`, the verify pass) are committed
in four passes; when one fails to encode, the passes before it have run,
so the state is partly advanced although its position is not: discard it
or restore a snapshot taken before the step.

## Differences from ds4

- The MTP head attends at most the last 2 051 rows of its run (ds4: the whole
  run). This changes the drafts, not the sampling rule: drafts only decide
  which steps verify two rows (see the finite-precision caveat above).
- Prefill GEMMs round both operands to bfloat16 (the weights to nearest,
  ties away from zero; f32 accumulators); decode works on f32 activations.
  Quality is close to ds4's (below).
- Prompt runs of more than 32 rows (gate bound in [-5, 0]) compute KDA in
  its chunked (WY) form, 32 tokens a chunk, in f32: algebraically the
  recurrence, computed from cumulative log decays and factored
  exponentials, and checked against it within the tests' tolerances, not
  bit for bit. Decode and verify rows step the recurrence. Quality is close to ds4's (below).
- Prompt runs compute DSA attention on the matrix units: the f32 queries
  and attention weights are split into an f16 high part (the value with its
  mantissa cut to ten bits) and the f16 rounding of the rest, multiplied
  with the f16 latent rows into f32 accumulators. The split keeps about 22
  significant bits while the rest is in f16's normal range (values from
  about 0.06 up); below that its error is bounded in absolute terms, about
  2^-24 (6e-8): 1e-4 comes back with a relative error of 1.7e-4. A score
  tile whose queries leave f16's range (or are NaN) is computed in f32.
  `LILY_GLM53_DSA_F32` (set to anything) keeps prompt runs on the f32
  kernels. Decode and verify rows compute attention in f32.
- The hyper-connection control projection scales its sums by the RMS after
  the dot products (ds4 normalizes first): the products differ in rounding
  only.

## Tests and tools

```sh
cargo test --release --lib -- glm53 gguf    # kernel tests need a GPU, no file
LILY_GLM53_GGUF=<path> cargo test --release --lib glm53 -- --ignored --test-threads=1
```

The ignored tests load the real file: the header against the schema,
speculative steps against ordinary decoding (accepted and rejected drafts,
rollback across an indexer pool past the sparse window, snapshot restore and
fork), sampled speculation against ordinary sampling (the same draws and
penalty counts after a rejection and an acceptance, and following the
engine's own proposals with and without a draft), the draft threshold, a
step that fails after its first passes, and cancelled prefill. `lily-glm53` (`src/bin/lily-glm53.rs`) compares
the GPU with the CPU reference layer by layer (`compare`), scores ds4's
official GLM 5.3 Flash continuations from its `gguf-tools/quality-testing`
(`score`, `--legacy` for ds4's reference rendering), measures prefill,
decode and speculative decoding (`bench`, `spec`), and locates where
speculative decoding first parts from plain decoding (`spec-diff`).

Measured on the M5 Max, 2026-11-13: ds4's 100 official cases NLL 0.466 per
token and 89/100 first tokens (ds4's own references 0.458 to 0.462 and 89 to
90); its 8 long cases (3K to 7K tokens) 0.666 and 6/8 (ds4 0.662 to 0.680,
6/8). Decode about 26 tok/s; speculative 27 to 37 tok/s depending on how
predictable the text is (greedy runs, which gave plain decoding's tokens);
prefill 220 tok/s at 1K tokens and 160 at 4K.

## Not implemented

Batching across sessions (the server decodes one request at a time), sessions
saved to disk, vision, Metal kernels.
