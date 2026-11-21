# Inkling 2 Mini

lily serves Thinking Machines Lab's **Inkling 2 Mini** (26.4B total parameters,
~3.4B–5.4B active: 24 physical layers with layers 6–17 looped 3 times for 48
logical layer visits, 128 routed experts top 8 with SITU activation,
relative-position attention with zero sink and causal short convolutions) from
published GGUF files on the Linux/Vulkan backend (both M5 matrix-unit GPUs and
M4 / non-matrix-unit GPUs).

Both mainline `inkling2_loop` GGUFs (`ggml-org/Inkling-2-Mini-GGUF` and
`unsloth/Inkling-2-Mini-GGUF/Mainline_Support/`) and Unsloth's root `inkling2`
looped GGUFs load directly without conversion.

## Files for 16 GB, 32 GB, 64 GB and 128 GB machines

| Machine | Repository | File | Size | Notes |
|---|---|---|---|---|
| 16 GB | `unsloth/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-UD-IQ2_XXS.gguf` | 8.49 GB | Fits `--memory-limit-gb 10.2` with room for context |
| 16 GB | `unsloth/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-UD-Q2_K_XL.gguf` | 9.14 GB | Higher-quality 2-bit dynamic quant for 16 GB |
| 32 GB | `ggml-org/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-IQ4_NL.gguf` | 17.63 GB | 4-bit layers 0–19 experts, Q8_0 layers 20–23 & attention |
| 32 GB | `unsloth/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-UD-Q4_K_XL.gguf` | 17.63 GB | Unsloth dynamic 4-bit quant |
| 64 GB | `ggml-org/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-Q8_0.gguf` | 27.82 GB | Near-lossless 8-bit |
| 128 GB | `ggml-org/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-BF16.gguf` | 52.23 GB | Full BF16 reference precision |

Optional add-ons (automatically discovered beside the model GGUF):

- **DSpark speculative drafter** (`unsloth/Inkling-2-Mini-GGUF`):
  `dspark-Inkling-2-Mini-Q8_0.gguf` (0.40 GB, default) or
  `dspark/dspark-Inkling-2-Mini-BF16.gguf` (0.65 GB). Enabled with
  `--mtp-drafts 3` (up to 7).
- **Vision tower** (`mmproj`): `src/inkling2/vision.rs` implements the
  `inkling2v` ViT + projector (`mmproj-F16.gguf` / `mmproj-BF16.gguf` /
  `mmproj-Inkling-2-Mini-Q8_0.gguf`); HTTP image-request routing for 20x20
  patches and Inkling 2 placeholders is a follow-up.

## Running it

```sh
# 16 GB (M4 Air / low-RAM): UD-Q2_K_XL or UD-IQ2_XXS
./target/release/lily --model ~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf \
  --memory-limit-gb 10.2 --mtp-drafts 0 --checkpoints-per-session 1 \
  --decode-checkpoint-tokens 0 --pin-weights off

# 32 GB: IQ4_NL (or UD-Q4_K_XL) with DSpark speculative decoding
./target/release/lily --model ~/models/Inkling-2-Mini-IQ4_NL.gguf \
  --memory-limit-gb 22 --mtp-drafts 3 --decode-checkpoint-tokens 0 \
  --pin-weights off

# 64 GB / 128 GB: full Q8_0 or BF16 with DSpark + vision mmproj beside it
./target/release/lily --model ~/models/Inkling-2-Mini-Q8_0.gguf \
  --memory-limit-gb 44 --mtp-drafts 3 --pin-weights off
```

## Architecture & kernels

- `src/inkling2/`: `config` (parses both `inkling2_loop` and Unsloth `inkling2`),
  `schema` (validates every tensor's shape and GGML block type, including
  optional per-expert NVFP4/MXFP4 `.scale` tensors), `reference` (f32 CPU
  reference for every op), `model` (48-slot KV + short-conv state, looped core),
  `dspark` (3-layer DFlare drafter + `rnn_hidden` Markov head), `vision`
  (23-layer `inkling2v` ViT with 2D RoPE, LayerScale, 2x2 merge + GELU-ERF
  projector), `tokenizer` (GPT-4o / o200k BPE + chat template), `engine`
  (`LanguageModel` + `--memory-limit-gb` fit).
- `src/kernels/inkling2.rs` and `src/kernels/vulkan/inkling2.comp`:
  RMSNorm, per-head L2 clip with log-scale `tau`, block attention-residual mix,
  rank-64 cross-pass loop controllers, static and dynamic causal short
  convolutions, relative-position bias GQA attention with implicit zero sink,
  sigmoid top-8 router with bias & `route_scale`, SITU activation, and generic
  GGML embedding/MoE kernels covering all 19 GGML types (including `NVFP4`,
  `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_XS`, `MXFP4`, `IQ4_NL`, `Q8_0`,
  `BF16`).
- `lily-inkling2`: bring-up, profiling, and benchmark tool (`fit`, `chat`, `bench`, `profile`).

## Measured throughput (16 GB M4 Air, Linux 7.3.0 / Borealis)

- `Inkling-2-Mini-UD-Q2_K_XL.gguf` (9.14 GB) + `dspark-Inkling-2-Mini-Q8_0.gguf` (0.40 GB)
  with `--memory-limit-gb 10.6 --mtp-drafts 3 --checkpoints-per-session 1 --decode-checkpoint-tokens 0 --pin-weights off`
  (11.4 GB on the GPU, up to 37 852 tokens of context with 1 snapshot and DSpark):
  **17.8 tok/s** end-to-end decode (73 generated tokens, 50/66 drafts accepted, 19.5 tok/s GPU steady-state,
  identical greedy token IDs to plain decode), **13.0 tok/s** plain decode (`77.2 ms/tok`),
  **49.9 tok/s** prefill (256 tokens, 64-row chunks).
