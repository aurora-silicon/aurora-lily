// The register path of glm53_gemm.comp's expert GEMMs for a tile of
// GM_DR rows (included once per row count inside the kernel's main: its
// variables are main's). The decode is the tile path's; each thread dots
// its own decoded weights with the rows' bfloat values, and the eight
// threads of a weight row add their sums through threadgroup memory in k8
// order.
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
        [[unroll]] for (uint r = 0u; r < 4u; r++) gm_decode8(wr[r], k0 + 8u * k8, w[r]);
        [[unroll]] for (uint j = 0u; j < GM_DR; j++) {
            const uvec4 v = xd.v[xr[j] + (k0 >> 3u) + k8];
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
    // Partial sums: word (k8 * 64 + col) * GM_DR + j.
    [[unroll]] for (uint r = 0u; r < 4u; r++)
        [[unroll]] for (uint j = 0u; j < GM_DR; j++) {
            const uint f = (k8 * 64u + 4u * quad + r) * GM_DR + j;
            gm_tile[f >> 2u][f & 3u] = floatBitsToUint(dacc[r][j]);
        }
    barrier();
    // Thread t sums column t % 64 of row t / 64 (and t + 128 ..).
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
#ifdef GM_GATE_UP
    // Gate's sums (columns 0 .. 32) meet up's (32 .. 64) in the tile's
    // second half.
    const uint base = GM_STEP * 8u;
    [[unroll]] for (uint h = 0u; h < (64u * GM_DR + 127u) / 128u; h++) {
        const uint t = tid + 128u * h;
        if (t < 64u * GM_DR) gm_tile[base + (t >> 2u)][t & 3u] = floatBitsToUint(sums[h]);
    }
    barrier();
    const float limit = arg_f32(12);
    if (tid < 32u * GM_DR) {
        const uint c = tid & 31u, j = tid >> 5u;
        if (j < rows) {
            const uint gi = j * 64u + c, ui = j * 64u + 32u + c;
            float g = uintBitsToFloat(gm_tile[base + (gi >> 2u)][gi & 3u]);
            float u = uintBitsToFloat(gm_tile[base + (ui >> 2u)][ui & 3u]);
            if (limit > 1e-6) {
                g = min(g, limit);
                u = clamp(u, -limit, limit);
            }
            const uint p = U32(arg(3)).v[first + j];
            const float we = F32(arg(7)).v[p];
            precise float gu = g * u;
            precise float e = exp2(-(g * uintBitsToFloat(0x3fb8aa3bu)));
            precise float sv = we * (gu * (1.0 / (1.0 + e)));
            BF16(arg(2) + 2u * (uint64_t(p) * N + n0 + c)).v[0] =
                uintBitsToBFloat16EXT(uint16_t(glm53_bf16_bits(sv)));
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
