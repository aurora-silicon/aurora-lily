// sgemm.comp's tile, for the shape the includer defines: SGT_NAME(n0, rows,
// W, N, K, row_bytes) computes Y's rows 0 .. rows (< SGT_TR) x weight rows
// n0 .. n0 + SGT_TW (those below N) with 256 threads, K steps of SGT_KS
// (SGT_TW x SGT_KS = 4096 weights a step). Each K step decodes the tile's
// weights (a thread 16 values of a row) into sg_w beside the activations in
// sg_x, both transposed ([k][n / 4], [k][m / 4]) so a thread reads vectors
// of each; thread (tm, tn) = (tid / (SGT_TW / 8), tid % (SGT_TW / 8))
// computes rows 4 tm .. x weight rows 8 tn .. .
void SGT_NAME(uint n0, uint rows, uint64_t W, uint N, uint K, uint row_bytes) {
    const uint tid = gl_LocalInvocationIndex;
    // Weight decode: values dk .. dk + 16 of a step's SGT_KS of row dr.
    const uint dr = tid / (SGT_KS / 16u), dk = 16u * (tid % (SGT_KS / 16u));
    const uint64_t w_row = W + uint64_t(min(dr, N - 1u - n0)) * row_bytes;
    // Activation loads: vec4 q = tid + 256 h of a step's SGT_TR x SGT_KS
    // values, (row, 4 values) = (q / (SGT_KS / 4), q % (SGT_KS / 4)).
#define SGT_XQ (SGT_TR * SGT_KS / 4u)
    uint64_t x_at[(SGT_XQ + 255u) / 256u];
    [[unroll]] for (uint h = 0u; h < (SGT_XQ + 255u) / 256u; h++) {
        const uint q = min(tid + 256u * h, SGT_XQ - 1u);
        x_at[h] = arg(0) + 4u * (sg_x_row(min(q / (SGT_KS / 4u), rows - 1u)) + 4u * (q % (SGT_KS / 4u)));
    }
    const uint tn = tid % (SGT_TW / 8u), tm = tid / (SGT_TW / 8u);
    vec4 acc[4][2];
    [[unroll]] for (uint i = 0u; i < 4u; i++) acc[i][0] = acc[i][1] = vec4(0.0);
    for (uint k0 = 0u; k0 < K; k0 += SGT_KS) {
        [[unroll]] for (uint h = 0u; h < 2u; h++) {
            float w[8];
            sg_decode8(w_row, k0 + dk + 8u * h, w);
            [[unroll]] for (uint k = 0u; k < 8u; k++)
                sg_w[(dk + 8u * h + k) * (SGT_TW / 4u) + (dr >> 2u)][dr & 3u] = w[k];
        }
        [[unroll]] for (uint h = 0u; h < (SGT_XQ + 255u) / 256u; h++) {
            const uint q = tid + 256u * h, r = q / (SGT_KS / 4u), kq = q % (SGT_KS / 4u);
            if (q < SGT_XQ) {
                const vec4 v = F32x4(x_at[h] + 4u * k0).v[0];
                [[unroll]] for (uint i = 0u; i < 4u; i++)
                    sg_x[(4u * kq + i) * (SGT_TR / 4u) + (r >> 2u)][r & 3u] = v[i];
            }
        }
        barrier();
        // Threads past the tile's rows skip the products (whole subgroups
        // on the 32-row tile, pairs of them on the 64-row one): expert
        // tiles are often part full.
        if (4u * tm < rows) {
            [[unroll]] for (uint k = 0u; k < SGT_KS; k++) {
                const vec4 xv = sg_x[k * (SGT_TR / 4u) + tm];
                const vec4 w0 = sg_w[k * (SGT_TW / 4u) + 2u * tn];
                const vec4 w1 = sg_w[k * (SGT_TW / 4u) + 2u * tn + 1u];
                [[unroll]] for (uint i = 0u; i < 4u; i++) {
                    acc[i][0] = fma(vec4(xv[i]), w0, acc[i][0]);
                    acc[i][1] = fma(vec4(xv[i]), w1, acc[i][1]);
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint c = 0u; c < 2u; c++) {
        const uint n = n0 + 8u * tn + 4u * c;
        if (n >= N) return;
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            const uint r = 4u * tm + i;
            if (r < rows) {
                const uint64_t y = arg(2) + 4u * (sg_y_row(r) + n);
                if (n + 4u <= N) {
                    F32x4(y).v[0] = acc[i][c];
                } else {
                    for (uint j = 0u; j < N - n; j++) F32(y).v[j] = acc[i][c][j];
                }
            }
        }
    }
}
#undef SGT_XQ
#undef SGT_NAME
#undef SGT_TR
#undef SGT_TW
#undef SGT_KS
