// sgemm.comp's tile, for the shape the includer defines: SGT_NAME(n0, rows,
// W, N, K, row_bytes) computes Y's rows 0 .. rows (< SGT_TR) x weight rows
// n0 .. n0 + SGT_TW (those below N) with 256 threads, K steps of SGT_KS
// (SGT_TW x SGT_KS = 4096 weights a step).
//
// Both decoded weights (sg_w) and clamped activations (sg_x) are staged in
// threadgroup memory as f16vec4 with padded strides to avoid bank conflicts.
// Each thread computes an 8 x 4 output sub-tile using native FP16 FMAs over
// the K step (2x ALU throughput and ~2x FLOP/W on Apple G16/M4, critical on
// fanless thermal budgets), then widens the step sum into an f32 accumulator.
void SGT_NAME(uint n0, uint rows, uint64_t W, uint N, uint K, uint row_bytes) {
    const uint tid = gl_LocalInvocationIndex, sg = tid >> 5u, lane = tid & 31u;
    // Weight decode: values dk .. dk + 16 of a step's SGT_KS of row dr.
    const uint dr = tid / (SGT_KS / 16u), dk = 16u * (tid % (SGT_KS / 16u));
    const uint64_t w_row = W + uint64_t(min(dr, N - 1u - n0)) * row_bytes;
    const uint mg = sg % (SGT_TR / 8u), ng = sg / (SGT_TR / 8u);
    const uint m_lo = 8u * mg;
    const uint w_stride = (SGT_TW / 4u) + 1u;
    const uint x_stride = (SGT_TR / 4u) + 1u;
#if SGT_TR == 64 && SGT_KS == 32
    // 256 threads load 64 x 32 = 2048 X elements -> 8 elements (2 vec4) per thread.
    const uint xm = tid >> 2u;
    const uint xk = (tid & 3u) * 8u;
    const uint64_t x_load = arg(0) + 4u * (sg_x_row(min(xm, rows - 1u)) + xk);
#else
    // 32 x 16 = 512 X elements -> first 128 threads load 4 elements (1 vec4) each.
    const uint xm = (tid & 127u) >> 2u;
    const uint xk = (tid & 3u) * 4u;
    const uint64_t x_load = arg(0) + 4u * (sg_x_row(min(xm, rows - 1u)) + xk);
#endif
    vec4 acc[8];
    [[unroll]] for (uint i = 0u; i < 8u; i++) acc[i] = vec4(0.0);
#if SGT_KS == 16
    f16vec4 a16[8];
    [[unroll]] for (uint i = 0u; i < 8u; i++) a16[i] = f16vec4(0.0);
#endif
    for (uint k0 = 0u; k0 < K; k0 += SGT_KS) {
        [[unroll]] for (uint h = 0u; h < 2u; h++) {
            float w[8];
            sg_decode8(w_row, k0 + dk + 8u * h, w);
            [[unroll]] for (uint k = 0u; k < 8u; k++) {
                sg_w[(dk + 8u * h + k) * w_stride + (dr >> 2u)][dr & 3u] = float16BitsToUint16(float16_t(w[k]));
            }
        }
#if SGT_TR == 64 && SGT_KS == 32
        const f16vec4 xv0 = f16vec4(clamp(F32x4(x_load + 4u * k0).v[0], vec4(-65504.0), vec4(65504.0)));
        const f16vec4 xv1 = f16vec4(clamp(F32x4(x_load + 4u * (k0 + 4u)).v[0], vec4(-65504.0), vec4(65504.0)));
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            sg_x[(xk + k) * x_stride + (xm >> 2u)][xm & 3u] = xv0[k];
            sg_x[(xk + 4u + k) * x_stride + (xm >> 2u)][xm & 3u] = xv1[k];
        }
#else
        if (tid < 128u) {
            const f16vec4 xv = f16vec4(clamp(F32x4(x_load + 4u * k0).v[0], vec4(-65504.0), vec4(65504.0)));
            [[unroll]] for (uint k = 0u; k < 4u; k++) {
                sg_x[(xk + k) * x_stride + (xm >> 2u)][xm & 3u] = xv[k];
            }
        }
#endif
        barrier();
        // Subgroups whose rows are past the tile's skip the products:
        // expert tiles are often part full.
        if (m_lo < rows) {
#if SGT_KS == 32
            f16vec4 a16[8];
            [[unroll]] for (uint i = 0u; i < 8u; i++) a16[i] = f16vec4(0.0);
#else
            f16vec4 prev16[8];
            [[unroll]] for (uint i = 0u; i < 8u; i++) prev16[i] = a16[i];
#endif
            [[unroll]] for (uint k = 0u; k < SGT_KS; k++) {
                const f16vec4 w = uint16BitsToFloat16(sg_w[k * w_stride + 32u * ng + lane]);
                const f16vec4 xa = sg_x[k * x_stride + 2u * mg];
                const f16vec4 xb = sg_x[k * x_stride + 2u * mg + 1u];
                a16[0] = fma(f16vec4(xa.x), w, a16[0]);
                a16[1] = fma(f16vec4(xa.y), w, a16[1]);
                a16[2] = fma(f16vec4(xa.z), w, a16[2]);
                a16[3] = fma(f16vec4(xa.w), w, a16[3]);
                a16[4] = fma(f16vec4(xb.x), w, a16[4]);
                a16[5] = fma(f16vec4(xb.y), w, a16[5]);
                a16[6] = fma(f16vec4(xb.z), w, a16[6]);
                a16[7] = fma(f16vec4(xb.w), w, a16[7]);
            }
            bool ovf = false;
            [[unroll]] for (uint i = 0u; i < 8u; i++) {
                ovf = ovf || any(isinf(a16[i])) || any(isnan(a16[i]));
            }
            if (ovf) {
#if SGT_KS == 16
                if ((k0 & 16u) != 0u) {
                    [[unroll]] for (uint i = 0u; i < 8u; i++) acc[i] += vec4(prev16[i]);
                }
#endif
                const uint col_base = 128u * ng + 4u * lane;
                for (uint c = 0u; c < 4u; c++) {
                    const uint64_t cw = W + uint64_t(min(col_base + c, N - 1u - n0)) * row_bytes;
                    for (uint q = 0u; q < (SGT_KS / 8u); q++) {
                        float wc[8];
                        sg_decode8(cw, k0 + 8u * q, wc);
                        for (uint i = 0u; i < 8u; i++) {
                            const uint64_t xp = arg(0) + 4u * (sg_x_row(min(m_lo + i, rows - 1u)) + k0 + 8u * q);
                            const vec4 x0 = F32x4(xp).v[0];
                            const vec4 x1 = F32x4(xp + 16u).v[0];
                            float s = acc[i][c];
                            s = fma(wc[0], x0.x, s); s = fma(wc[1], x0.y, s);
                            s = fma(wc[2], x0.z, s); s = fma(wc[3], x0.w, s);
                            s = fma(wc[4], x1.x, s); s = fma(wc[5], x1.y, s);
                            s = fma(wc[6], x1.z, s); s = fma(wc[7], x1.w, s);
                            acc[i][c] = s;
                        }
                    }
                }
#if SGT_KS == 16
                [[unroll]] for (uint i = 0u; i < 8u; i++) a16[i] = f16vec4(0.0);
#endif
            } else {
#if SGT_KS == 32
                [[unroll]] for (uint i = 0u; i < 8u; i++) acc[i] += vec4(a16[i]);
#else
                if ((k0 & 16u) != 0u) {
                    [[unroll]] for (uint i = 0u; i < 8u; i++) {
                        acc[i] += vec4(a16[i]);
                        a16[i] = f16vec4(0.0);
                    }
                }
#endif
            }
        }
        barrier();
    }
    const uint n = n0 + 128u * ng + 4u * lane;
    if (n >= N) return;
    [[unroll]] for (uint i = 0u; i < 8u; i++) {
        const uint r = m_lo + i;
        if (r < rows) {
            const uint64_t y = arg(2) + 4u * (sg_y_row(r) + n);
            if (n + 4u <= N) {
                F32x4(y).v[0] = acc[i];
            } else {
                for (uint j = 0u; j < N - n; j++) F32(y).v[j] = acc[i][j];
            }
        }
    }
}
#undef SGT_NAME
#undef SGT_TR
#undef SGT_TW
#undef SGT_KS
