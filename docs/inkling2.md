# Inkling 2 Mini

lily serves Thinking Machines Lab's **Inkling 2 Mini** (26.4B total parameters,
~3.4B–5.4B active: 24 physical layers with layers 6–17 looped 3 times for 48
logical layer visits, 128 routed experts top 8 with SITU activation,
relative-position attention with zero sink and causal short convolutions) from
published GGUF files on the Linux/Vulkan backend (both M5 matrix-unit GPUs and
M4 / non-matrix-unit GPUs).

Both native Thinking Machines safetensors directories (`thinkingmachines/Inkling-2-Mini`
— BF16 dense weights, NVFP4 layers 0–19 experts, MXFP8 layers 20–23 experts, and
`mtp.safetensors` DSpark drafter) and published GGUFs (`ggml-org/Inkling-2-Mini-GGUF`
mainline `inkling2_loop` and Unsloth's root `inkling2` looped GGUFs) load directly
without external conversion.

## Files for 16 GB, 32 GB, 64 GB and 128 GB machines

| Machine | Repository | File / Directory | Size | Notes |
|---|---|---|---|---|
| 16 GB | `unsloth/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-UD-IQ2_XXS.gguf` | 8.49 GB | Fits `--memory-limit-gb 10.2` with room for context |
| 16 GB | `unsloth/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-UD-Q2_K_XL.gguf` | 9.14 GB | Higher-quality 2-bit dynamic quant for 16 GB |
| 32 GB | `ggml-org/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-IQ4_NL.gguf` | 17.63 GB | 4-bit layers 0–19 experts, Q8_0 layers 20–23 & attention |
| 32 GB | `unsloth/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-UD-Q4_K_XL.gguf` | 17.63 GB | Unsloth dynamic 4-bit quant |
| 32–64 GB | `thinkingmachines/Inkling-2-Mini` | HF checkpoint directory | 19.6 GB | Native NVFP4 (L0–19) + MXFP8 (L20–23) + BF16 + `mtp.safetensors`; ~19.9 GB resident |
| 64 GB | `ggml-org/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-Q8_0.gguf` | 27.82 GB | Near-lossless 8-bit |
| 128 GB | `ggml-org/Inkling-2-Mini-GGUF` | `Inkling-2-Mini-BF16.gguf` | 52.23 GB | Full BF16 reference precision |

Optional add-ons (automatically discovered beside the model GGUF, or inside the native HF directory):

- **DSpark speculative drafter**: `mtp.safetensors` (0.61 GB, bundled in
  `thinkingmachines/Inkling-2-Mini`) or GGUF drafters from
  `unsloth/Inkling-2-Mini-GGUF` (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.40 GB,
  default, or `dspark/dspark-Inkling-2-Mini-BF16.gguf`, 0.65 GB). Enabled with
  `--mtp-drafts 3` (up to 7). When the main LM head is BF16, lily builds a
  compact 261 MB `Q4_0` copy used only for DSpark drafting while keeping exact
  BF16 `output.weight` for speculative verification.
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

# 32 GB / 64 GB / 128 GB: native Thinking Machines HF directory (NVFP4 + MXFP8 + BF16 + mtp.safetensors)
./target/release/lily --model ~/models/Inkling-2-Mini-native \
  --memory-limit-gb 44 --mtp-drafts 3 --pin-weights off

# 64 GB / 128 GB: full Q8_0 or BF16 GGUF with DSpark + vision mmproj beside it
./target/release/lily --model ~/models/Inkling-2-Mini-Q8_0.gguf \
  --memory-limit-gb 44 --mtp-drafts 3 --pin-weights off
```

## Architecture & kernels

- `src/inkling2/`: `config` (parses HF `config.json`, mainline `inkling2_loop`,
  and Unsloth `inkling2`), `native` (direct `thinkingmachines/Inkling-2-Mini`
  safetensors + `mtp.safetensors` loader with lossless parallel NVFP4 repack and
  MXFP8→Q8_0 conversion), `schema` (validates every tensor's shape and GGML
  block type, including optional per-expert NVFP4/MXFP4 `.scale` tensors),
  `reference` (f32 CPU reference for every op), `model` (48-slot KV + short-conv
  state, looped core), `dspark` (3-layer DFlare drafter + `rnn_hidden` Markov
  head), `vision` (23-layer `inkling2v` ViT with 2D RoPE, LayerScale, 2x2 merge
  + GELU-ERF projector), `tokenizer` (GPT-4o / o200k BPE + chat template),
  `engine` (`LanguageModel` + `--memory-limit-gb` fit).
- `src/kernels/inkling2.rs`, `src/kernels/vulkan/inkling2.comp`,
  `src/kernels/vulkan/inkling2_attn.comp`, and `src/kernels/vulkan/inkling2_gemm.comp`:
  RMSNorm, per-head L2 clip with log-scale `tau`, vectorized block
  attention-residual mix with optional fused RMSNorm, rank-64 F32/BF16 cross-pass
  loop controllers, static and dynamic causal short convolutions (parallel
  prefill + sequential decode/verify), tiled relative-position bias, Split-K GQA
  attention (decode/verify + DSpark), matrix-unit FlashAttention
  (`inkling2_attn_mu`, `AD=128`, `BQ=16`, `BK=128`) with implicit zero sink and
  fused sigmoid gate, subgroup top-8 sigmoid router with bias & `route_scale`,
  SITU activation, constant-cache LUT NVFP4 & specialized `cnt=1..4` fused
  decode/verify MoE kernels, and matrix-unit grouped MoE kernels
  (`inkling2_gemm_moe_gu_situ_*`, `inkling2_gemm_moe_down_*`) covering all 19
  GGML types (including `NVFP4`, `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_XS`,
  `MXFP4`, `IQ4_NL`, `Q8_0`, `BF16`).
- `lily-inkling2`: bring-up, profiling (`--kernel-profile`, `--ctx-len`), and
  benchmark tool (`fit`, `chat`, `bench`, `profile`).

## Measured throughput

### M5 Max (128 GB `j716c`, Linux / Borealis, matrix units)

- Native `thinkingmachines/Inkling-2-Mini` (19.9 GB resident, 4.1 s load,
  bundled `mtp.safetensors` + Q4_0 draft LM head):
  - **Prefill**: **2,420 tok/s** at 1,024 tokens, **2,361 tok/s** at 4,096 tokens
    (1.74 s), **2,122 tok/s** end-to-end on the 5,318-token `served5.py`
    `long-context` prompt (2.50 s TTFT).
  - **Decode**: **49.6 tok/s** plain decode (`--mtp-drafts 0`), **83.6 tok/s**
    at `--mtp-drafts 3` and **87.2 tok/s** at `--mtp-drafts 5` (1,024-token
    prompt), **59.3 tok/s** at 4,096-token context (`--mtp-drafts 3`), and
    **54.6–78.3 tok/s** across the 5-prompt `served5.py` suite (**78.3 tok/s**
    on `code`, **60.0 tok/s** on 5,318-token `long-context`).
- `Inkling-2-Mini-Q8_0.gguf` (27.81 GB) + `dspark-Inkling-2-Mini-Q8_0.gguf` (0.40 GB):
  - **Prefill**: **2,291 tok/s** at 1,024 tokens, **2,214 tok/s** at 4,096 tokens
    (1.85 s), **1,946 tok/s** on the 5,321-token `served5.py` `long-context` prompt.
  - **Decode**: **52.4 tok/s** plain decode (`--mtp-drafts 0`), **80.5 tok/s**
    at `--mtp-drafts 3` and **81.7 tok/s** at `--mtp-drafts 5` (1,024-token
    prompt), **59.9 tok/s** at 4,096-token context (`--mtp-drafts 3`).

### 16 GB M4 Air (Linux 7.3.0 / Borealis)

- `Inkling-2-Mini-UD-Q2_K_XL.gguf` (9.14 GB) + `dspark-Inkling-2-Mini-Q8_0.gguf` (0.40 GB)
  with `--memory-limit-gb 10.6 --mtp-drafts 3 --checkpoints-per-session 1 --decode-checkpoint-tokens 0 --pin-weights off`
  (11.4 GB on the GPU, up to 37 852 tokens of context with 1 snapshot and DSpark):
  **17.8 tok/s** end-to-end decode (73 generated tokens, 50/66 drafts accepted, 19.5 tok/s GPU steady-state,
  identical greedy token IDs to plain decode), **13.0 tok/s** plain decode (`77.2 ms/tok`),
  **49.9 tok/s** prefill (256 tokens, 64-row chunks).
