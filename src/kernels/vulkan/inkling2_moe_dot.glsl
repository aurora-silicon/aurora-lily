// Register dot path of inkling2_gemm.comp for small expert tiles (GM_DR in 1..=4).
// Mirrors glm53_moe_dot.glsl, with SITU activation + optional per-expert scale
// for I2_GM_GATE_UP (writing BF16 mid) and F32 output for down.
{
    U32x4 xd = U32x4(arg(0));
    uint xr[GM_DR];
    [[unroll]] for (uint j = 0u; j < GM_DR; j++) {
        const uint pr = U32(arg(3)).v[first + min(j, rows - 1u)];
        xr[j] = (x_shift >= 0x80000000u ? pr / (x_shift & 0x7fffffffu) : pr >> x_shift) * (K / 8u);
    }
    float dacc[4][GM_DR];
    [[unroll]] for (uint r = 0u; r < 4u; r++)
        [[unroll]] for (uint j = 0u; j < GM_DR; j++) dacc[r][j] = 0.0;
    for (uint k0 = 0u; k0 < K; k0 += GM_STEP) {
        float w[4][8];
        [[unroll]] for (uint r = 0u; r < 4u; r++) i2_gm_decode8(dwr[r], k0 + 8u * dk8, w[r]);
        [[unroll]] for (uint j = 0u; j < GM_DR; j++) {
            const uvec4 v = xd.v[xr[j] + (k0 >> 3u) + dk8];
            float xf[8];
            [[unroll]] for (uint c = 0u; c < 4u; c++) {
                xf[2u * c] = uintBitsToFloat(v[c] << 16u);
                xf[2u * c + 1u] = uintBitsToFloat(v[c] & 0xffff0000u);
            }
            [[unroll]] for (uint r = 0u; r < 4u; r++) {
                [[unroll]] for (uint i = 0u; i < 8u; i++) dacc[r][j] = fma(w[r][i], xf[i], dacc[r][j]);
            }
        }
    }
    // Partial sums: word (dk8 * 64 + col) * GM_DR + j.
    [[unroll]] for (uint r = 0u; r < 4u; r++)
        [[unroll]] for (uint j = 0u; j < GM_DR; j++) {
            const uint f = (dk8 * 64u + 4u * dq + r) * GM_DR + j;
            gm_tile[f >> 2u][f & 3u] = floatBitsToUint(dacc[r][j]);
        }
    barrier();
    float sums[(64u * GM_DR + 127u) / 128u];
    [[unroll]] for (uint h = 0u; h < (64u * GM_DR + 127u) / 128u; h++) {
        const uint t = tid + 128u * h, c = t & 63u, j = t >> 6u;
        float s = 0.0;
        if (j < GM_DR) {
            [[unroll]] for (uint z = 0u; z < 8u; z++) {
                const uint f = (z * 64u + c) * GM_DR + j;
                s += uintBitsToFloat(gm_tile[f >> 2u][f & 3u]);
            }
        }
        sums[h] = s;
    }
#ifdef I2_GM_GATE_UP
    // Gate's sums (columns 0..32) meet up's (32..64) in the second half of gm_tile.
    const uint base = GM_STEP * 8u;
    [[unroll]] for (uint h = 0u; h < (64u * GM_DR + 127u) / 128u; h++) {
        const uint t = tid + 128u * h;
        if (t < 64u * GM_DR) gm_tile[base + (t >> 2u)][t & 3u] = floatBitsToUint(sums[h]);
    }
    barrier();
    const float sc = arg_u32(11) != 0u ? F32(arg(6)).v[expert] : 1.0;
    const float beta = arg_f32(12);
    const float lb = arg_f32(13);
    if (tid < 32u * GM_DR) {
        const uint c = tid & 31u, j = tid >> 5u;
        if (j < rows && n0 + c < N) {
            const uint gi = j * 64u + c, ui = j * 64u + 32u + c;
            const float g = uintBitsToFloat(gm_tile[base + (gi >> 2u)][gi & 3u]) * sc;
            const float u = uintBitsToFloat(gm_tile[base + (ui >> 2u)][ui & 3u]) * sc;
            const float act = beta * i2_tanh(g / beta) / (1.0 + exp(-g));
            const float lin = lb > 0.0 ? lb * i2_tanh(u / lb) : u;
            const uint p = U32(arg(3)).v[first + j];
            BF16(arg(2) + 2u * (uint64_t(p) * N + n0 + c)).v[0] =
                uintBitsToBFloat16EXT(uint16_t(i2_bf16_bits(act * lin)));
        }
    }
#else
    [[unroll]] for (uint h = 0u; h < (64u * GM_DR + 127u) / 128u; h++) {
        const uint t = tid + 128u * h, c = t & 63u, j = t >> 6u;
        if (j < rows && n0 + c < N) {
            F32(arg(2) + 4u * uint64_t(U32(arg(3)).v[first + j]) * N).v[n0 + c] = sums[h];
        }
    }
#endif
}
