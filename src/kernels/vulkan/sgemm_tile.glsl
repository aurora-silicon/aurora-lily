// sgemm.comp's tile, for the shape the includer defines: SGT_NAME(n0, rows,
// W, N, K, row_bytes) computes Y's rows 0 .. rows (< SGT_TR) x weight rows
// n0 .. n0 + SGT_TW (those below N) with 256 threads, K steps of SGT_KS
// (SGT_TW x SGT_KS = 4096 weights a step). Each K step decodes the tile's
// weights (a thread 16 values of a row) into sg_w, [k][n / 4]. Subgroup
// (mg, ng) computes rows 8 mg .. x weight rows 128 ng .., a lane the 8 rows
// x weight rows 128 ng + 4 lane ..: per k it reads its 4 weights (a vec4)
// and takes each row's activation from the lane that holds it
// (subgroupBroadcast), 32 FMAs. The lanes load the subgroup's activations
// straight into registers: lane l holds row 8 mg + l % 8, values 4 (l / 8)
// .. of each 16.
void SGT_NAME(uint n0, uint rows, uint64_t W, uint N, uint K, uint row_bytes) {
    const uint tid = gl_LocalInvocationIndex, sg = tid >> 5u, lane = tid & 31u;
    // Weight decode: values dk .. dk + 16 of a step's SGT_KS of row dr.
    const uint dr = tid / (SGT_KS / 16u), dk = 16u * (tid % (SGT_KS / 16u));
    const uint64_t w_row = W + uint64_t(min(dr, N - 1u - n0)) * row_bytes;
    const uint mg = sg % (SGT_TR / 8u), ng = sg / (SGT_TR / 8u);
    const uint m_lo = 8u * mg;
    const uint64_t x_at =
        arg(0) + 4u * (sg_x_row(min(m_lo + (lane & 7u), rows - 1u)) + 4u * (lane >> 3u));
    vec4 acc[8];
    [[unroll]] for (uint i = 0u; i < 8u; i++) acc[i] = vec4(0.0);
    for (uint k0 = 0u; k0 < K; k0 += SGT_KS) {
        [[unroll]] for (uint h = 0u; h < 2u; h++) {
            float w[8];
            sg_decode8(w_row, k0 + dk + 8u * h, w);
            [[unroll]] for (uint k = 0u; k < 8u; k++)
                sg_w[(dk + 8u * h + k) * (SGT_TW / 4u) + (dr >> 2u)][dr & 3u] = w[k];
        }
        vec4 xr[SGT_KS / 16u];
        [[unroll]] for (uint h = 0u; h < SGT_KS / 16u; h++) xr[h] = F32x4(x_at + 4u * (k0 + 16u * h)).v[0];
        barrier();
        // Subgroups whose rows are past the tile's skip the products:
        // expert tiles are often part full.
        if (m_lo < rows) {
            [[unroll]] for (uint k = 0u; k < SGT_KS; k++) {
                const vec4 w = sg_w[k * (SGT_TW / 4u) + 32u * ng + lane];
                [[unroll]] for (uint i = 0u; i < 8u; i++) {
                    const float x = subgroupBroadcast(xr[k / 16u][k % 4u], i + 8u * ((k % 16u) / 4u));
                    acc[i] = fma(vec4(x), w, acc[i]);
                }
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
