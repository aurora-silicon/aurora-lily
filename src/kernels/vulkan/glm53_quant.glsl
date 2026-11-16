// GGUF block decoders for GLM 5.3 Flash and Gemma 4 (src/gguf.rs is the CPU
// version; layouts and value formulas are ggml's as antirez's ds4, MIT, has
// them, and ggml's own, llama.cpp MIT, for Q4_0, Q5_K, Q6_K, IQ4_NL, IQ2_S).
// Every decoder turns one "task" of 32 consecutive values of a row into
// floats: one Q8_0, Q4_0 or IQ4_NL block, one 32-value sub-block of Q4_K,
// Q5_K or Q6_K, two 16-value groups of Q2_K, one 32-value group of IQ2_XXS
// or IQ2_S, or 32 BF16/F32 values. Each value is a product of small factors
// in ggml's order; where it is not exact in f32 (Q6_K) the order is pinned
// with `precise`.

#define GLM53_TASK 32u

// A product pinned as written (`precise`), declared as `name`; see
// common.glsl for the two forms.
#ifdef LILY_PRECISE_LEGACY
#define GLM53_PRECISE_MUL(name, a, b) precise float name = (a) * (b)
#else
float glm53_precise_mul(float a, float b) {
    precise float r = a * b;
    return r;
}
#define GLM53_PRECISE_MUL(name, a, b) const float name = glm53_precise_mul(a, b)
#endif

// Type ids (GGUF): the kernels take the type as a compile-time constant.
#define GLM53_F32 0u
#define GLM53_F16 1u
#define GLM53_Q4_0 2u
#define GLM53_Q8_0 8u
#define GLM53_Q2_K 10u
#define GLM53_Q4_K 12u
#define GLM53_Q5_K 13u
#define GLM53_Q6_K 14u
#define GLM53_IQ2_XXS 16u
#define GLM53_IQ4_NL 20u
#define GLM53_IQ2_S 22u
#define GLM53_BF16 30u

float glm53_h2f(uint bits) { return unpackHalf2x16(bits).x; }

// Bytes of a 32-value task inside a row: the tasks of one block are laid out
// at the block's offset; this gives the block's byte offset for task t.
// Q8_0 34-byte blocks of 32; K-quants 256-value blocks.

// --- Q8_0: f16 d, 32 int8. Block base is 2-byte aligned. ---
void glm53_decode_q8_0(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 34u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    [[unroll]] for (uint i = 0u; i < 16u; i++) {
        const uint pair = uint(U16(b + 2u).v[i]);
        w[2u * i] = d * float(int(pair << 24) >> 24);
        w[2u * i + 1u] = d * float(int(pair << 16) >> 24);
    }
}

// --- Q4_K: f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes (144-byte
// blocks, 16-byte aligned in these rows). Task t: block t / 8, sub-block
// j = t % 8: 6-bit scale and min, the low (even j) or high nibbles of bytes
// 32 * (j / 2).. ---
void glm53_decode_q4_k(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 144u;
    const uint j = t & 7u;
    const uint dd = U32(b).v[0];
    const float d = glm53_h2f(dd & 0xffffu);
    const float dmin = glm53_h2f(dd >> 16u);
    // Scale bytes q[0..12] as three words.
    const uvec3 sw = uvec3(U32(b + 4u).v[0], U32(b + 8u).v[0], U32(b + 12u).v[0]);
    #define Q4K_BYTE(i) ((sw[(i) >> 2u] >> (8u * ((i) & 3u))) & 0xffu)
    uint sc, m;
    if (j < 4u) {
        sc = Q4K_BYTE(j) & 63u;
        m = Q4K_BYTE(j + 4u) & 63u;
    } else {
        sc = (Q4K_BYTE(j + 4u) & 0xfu) | ((Q4K_BYTE(j - 4u) >> 6u) << 4u);
        m = (Q4K_BYTE(j + 4u) >> 4u) | ((Q4K_BYTE(j) >> 6u) << 4u);
    }
    #undef Q4K_BYTE
    const float scale = d * float(sc);
    const float minv = dmin * float(m);
    const uint shift = (j & 1u) * 4u;
    const uint64_t qs = b + 16u + (j >> 1u) * 32u;
    [[unroll]] for (uint q = 0u; q < 2u; q++) {
        const uvec4 v = U32x4(qs + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            const uint word = v[c] >> shift;
            [[unroll]] for (uint k = 0u; k < 4u; k++) {
                w[16u * q + 4u * c + k] = scale * float((word >> (8u * k)) & 0xfu) - minv;
            }
        }
    }
}

// --- Q2_K: 16 scale bytes (low nibble scale, high nibble min), 64 bytes of
// 2-bit codes, f16 d, f16 dmin (84-byte blocks, 4-byte aligned). Task t:
// block t / 8, groups g = 2 (t % 8) and g + 1, which read the 32 bytes at
// 32 * ((t % 8) / 4) at bit 2 * (t % 4). ---
void glm53_decode_q2_k(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 84u;
    const uint j = t & 7u;
    const uint dd = U32(b + 80u).v[0];
    const float d = glm53_h2f(dd & 0xffffu);
    const float dmin = glm53_h2f(dd >> 16u);
    const uint sc2 = uint(U16(b + 2u * j).v[0]);
    const uint shift = (j & 3u) * 2u;
    const uint64_t qs = b + 16u + (j >> 2u) * 32u;
    [[unroll]] for (uint h = 0u; h < 2u; h++) {
        const uint sc = (sc2 >> (8u * h)) & 0xffu;
        const float scale = d * float(sc & 0xfu);
        const float minv = dmin * float(sc >> 4u);
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            const uint word = U32(qs + 16u * h + 4u * c).v[0] >> shift;
            [[unroll]] for (uint k = 0u; k < 4u; k++) {
                w[16u * h + 4u * c + k] = scale * float((word >> (8u * k)) & 3u) - minv;
            }
        }
    }
}

// --- IQ2_XXS: f16 d, then per 32 values two words: four grid indices and
// four 7-bit sign indices plus a 4-bit scale (66-byte blocks, 2-byte
// aligned). The sign byte of index s is s with an even-parity eighth bit. ---
uint glm53_iq2xxs_grid[512] = uint[512](
#include "glm53_iq2xxs_grid.glsl"
);

void glm53_decode_iq2_xxs(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 66u;
    const uint ib = t & 7u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    const uint64_t q = b + 2u + 8u * ib;
    const uint aux0 = uint(U16(q).v[0]) | (uint(U16(q).v[1]) << 16u);
    const uint aux1 = uint(U16(q).v[2]) | (uint(U16(q).v[3]) << 16u);
    const float scale = 0.125 * d * float(2u * (aux1 >> 28u) + 1u);
    [[unroll]] for (uint l = 0u; l < 4u; l++) {
        const uint g = (aux0 >> (8u * l)) & 0xffu;
        const uint s7 = (aux1 >> (7u * l)) & 127u;
        const uint signs = s7 | ((bitCount(s7) & 1u) << 7u);
        const uint lo = glm53_iq2xxs_grid[2u * g];
        const uint hi = glm53_iq2xxs_grid[2u * g + 1u];
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            const float v = float(((k < 4u ? lo : hi) >> (8u * (k & 3u))) & 0xffu);
            w[8u * l + k] = scale * ((signs >> k & 1u) != 0u ? -v : v);
        }
    }
}

// Gemma 4's types, compiled only into the kernels that take them (a file
// defines GLM53_EXT_TYPES for those): GLM's kernels stay as they were.
#ifdef GLM53_EXT_TYPES
// --- Q4_0 and IQ4_NL: f16 d, 16 nibble bytes (18-byte blocks, 2-byte
// aligned); value j is the low nibble of byte j, value j + 16 its high
// nibble: Q4_0 (code - 8) * d, IQ4_NL d * kvalues_iq4nl[code]. ---
void glm53_decode_q4_0(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 18u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    [[unroll]] for (uint i = 0u; i < 8u; i++) {
        const uint pair = uint(U16(b + 2u).v[i]);
        [[unroll]] for (uint k = 0u; k < 2u; k++) {
            const uint byte = (pair >> (8u * k)) & 0xffu;
            w[2u * i + k] = float(int(byte & 0xfu) - 8) * d;
            w[2u * i + k + 16u] = float(int(byte >> 4u) - 8) * d;
        }
    }
}

// (dot, sum) of 32 values with 32 f32 activations at `x` (16-byte aligned).
vec2 glm53_task_dot_sum(in float w[32], uint64_t x) {
    float acc = 0.0, sx = 0.0;
    [[unroll]] for (uint q = 0u; q < 8u; q++) {
        const vec4 v = F32x4(x + 16u * q).v[0];
        acc += w[4u * q] * v.x + w[4u * q + 1u] * v.y + w[4u * q + 2u] * v.z + w[4u * q + 3u] * v.w;
        sx += (v.x + v.y) + (v.z + v.w);
    }
    return vec2(acc, sx);
}

// The IQ4_NL code value (ggml's kvalues_iq4nl): a select tree, not a dynamically indexed array
// (which the compiler copies to per-thread scratch).
float glm53_iq4nl_value(uint c) {
    // The sixteen codes as signed bytes, four to a word (little-endian).
    const uint w = c < 8u ? (c < 4u ? 0xBFAD9881u : 0xF6EADDCFu)
                          : (c < 12u ? 0x26190D01u : 0x71594535u);
    return float(bitfieldExtract(int(w), int(8u * (c & 3u)), 8));
}

void glm53_decode_iq4_nl(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 18u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    [[unroll]] for (uint i = 0u; i < 8u; i++) {
        const uint pair = uint(U16(b + 2u).v[i]);
        [[unroll]] for (uint k = 0u; k < 2u; k++) {
            const uint byte = (pair >> (8u * k)) & 0xffu;
            w[2u * i + k] = d * glm53_iq4nl_value(byte & 0xfu);
            w[2u * i + k + 16u] = d * glm53_iq4nl_value(byte >> 4u);
        }
    }
}

// The 6-bit scale and min of sub-block j of a Q4_K/Q5_K block at b (scale
// bytes at b + 4..16).
void glm53_k4_scale_min(uint64_t b, uint j, out uint sc, out uint m) {
    const uvec3 sw = uvec3(U16(b + 4u).v[0] | (uint(U16(b + 4u).v[1]) << 16u),
                           U16(b + 4u).v[2] | (uint(U16(b + 4u).v[3]) << 16u),
                           U16(b + 4u).v[4] | (uint(U16(b + 4u).v[5]) << 16u));
    #define K4_BYTE(i) ((sw[(i) >> 2u] >> (8u * ((i) & 3u))) & 0xffu)
    if (j < 4u) {
        sc = K4_BYTE(j) & 63u;
        m = K4_BYTE(j + 4u) & 63u;
    } else {
        sc = (K4_BYTE(j + 4u) & 0xfu) | ((K4_BYTE(j - 4u) >> 6u) << 4u);
        m = (K4_BYTE(j + 4u) >> 4u) | ((K4_BYTE(j) >> 6u) << 4u);
    }
    #undef K4_BYTE
}

// --- Q5_K: f16 d, f16 dmin, 12 scale bytes (Q4_K's), 32 bytes of high
// bits, 128 nibble bytes (176-byte blocks, 16-byte aligned in 2816-wide
// rows; read as 32-bit words). Task t: block t / 8, sub-block j = t % 8:
// Q4_K's nibbles plus bit j of high byte l as 16. ---
void glm53_decode_q5_k(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 176u;
    const uint j = t & 7u;
    const uint dd = U32(b).v[0];
    const float d = glm53_h2f(dd & 0xffffu);
    const float dmin = glm53_h2f(dd >> 16u);
    uint sc, m;
    glm53_k4_scale_min(b, j, sc, m);
    const float scale = d * float(sc);
    const float minv = dmin * float(m);
    const uint shift = (j & 1u) * 4u;
    const uint64_t qs = b + 48u + (j >> 1u) * 32u;
    [[unroll]] for (uint c = 0u; c < 8u; c++) {
        const uint word = U32(qs).v[c] >> shift;
        const uint hw = U32(b + 16u).v[c] >> j;
        [[unroll]] for (uint k = 0u; k < 4u; k++) {
            const uint q = ((word >> (8u * k)) & 0xfu) | (((hw >> (8u * k)) & 1u) << 4u);
            w[4u * c + k] = scale * float(q) - minv;
        }
    }
}

// --- Q6_K: 128 bytes of low nibbles, 64 of 2-bit highs, 16 int8 scales,
// f16 d (210-byte blocks, 2-byte aligned: read as 16-bit words). Task t:
// block t / 8, half n = (t % 8) / 4, quarter q = t % 4: value l takes nibble
// q / 2 of byte 64 n + 32 (q & 1) + l, bits 2 q of high byte 32 n + l,
// scale 8 n + 2 q + l / 16; value = (d * scale) * (code - 32). ---
void glm53_decode_q6_k(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 210u;
    const uint n = (t >> 2u) & 1u, q = t & 3u;
    const float d = glm53_h2f(uint(U16(b).v[104]));
    const uint sw = uint(U16(b + 192u).v[4u * n + q]);
    GLM53_PRECISE_MUL(s0, d, float(int(sw << 24) >> 24));
    GLM53_PRECISE_MUL(s1, d, float(int(sw << 16) >> 24));
    const uint64_t ql = b + 64u * n + 32u * (q & 1u);
    const uint64_t qh = b + 128u + 32u * n;
    const uint lshift = 4u * (q >> 1u), hshift = 2u * q;
    [[unroll]] for (uint i = 0u; i < 16u; i++) {
        const uint lo = uint(U16(ql).v[i]) >> lshift;
        const uint hi = uint(U16(qh).v[i]) >> hshift;
        [[unroll]] for (uint k = 0u; k < 2u; k++) {
            const uint c = ((lo >> (8u * k)) & 0xfu) | (((hi >> (8u * k)) & 3u) << 4u);
            GLM53_PRECISE_MUL(v, i < 8u ? s0 : s1, float(int(c) - 32));
            w[2u * i + k] = v;
        }
    }
}

// --- IQ2_S: f16 d, 32 grid-index low bytes, 32 sign bytes, 8 high-bit
// bytes, 8 scale bytes (82-byte blocks, 2-byte aligned). Task t: block
// t / 8, group ib = t % 8: values 8 l + j (l < 4) are grid vector qs[4 ib +
// l] | ((qh[ib] << (8 - 2 l)) & 0x300) times d (0.5 + scale nibble l / 2)
// / 4, negated by bit j of sign byte 4 ib + l. ---
uint glm53_iq2s_grid[2048] = uint[2048](
#include "glm53_iq2s_grid.glsl"
);

void glm53_decode_iq2_s(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 82u;
    const uint ib = t & 7u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    const uint qs = uint(U16(b + 2u).v[2u * ib]) | (uint(U16(b + 2u).v[2u * ib + 1u]) << 16u);
    const uint sg = uint(U16(b + 34u).v[2u * ib]) | (uint(U16(b + 34u).v[2u * ib + 1u]) << 16u);
    const uint hs = uint(U16(b + 66u).v[ib >> 1u]) >> (8u * (ib & 1u));
    const uint scb = uint(U16(b + 74u).v[ib >> 1u]) >> (8u * (ib & 1u));
    const float db0 = d * (0.5 + float(scb & 0xfu)) * 0.25;
    const float db1 = d * (0.5 + float((scb >> 4u) & 0xfu)) * 0.25;
    [[unroll]] for (uint l = 0u; l < 4u; l++) {
        const uint g = ((qs >> (8u * l)) & 0xffu) | (((hs & 0xffu) << (8u - 2u * l)) & 0x300u);
        const uint lo = glm53_iq2s_grid[2u * g];
        const uint hi = glm53_iq2s_grid[2u * g + 1u];
        const uint signs = (sg >> (8u * l)) & 0xffu;
        const float dl = l < 2u ? db0 : db1;
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            const float v = dl * float(((k < 4u ? lo : hi) >> (8u * (k & 3u))) & 0xffu);
            w[8u * l + k] = (signs >> k & 1u) != 0u ? -v : v;
        }
    }
}

#endif

// --- BF16, F16 and F32 rows (16-byte aligned here). ---
void glm53_decode_bf16(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 64u;
    [[unroll]] for (uint q = 0u; q < 4u; q++) {
        const uvec4 v = U32x4(b + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            w[8u * q + 2u * c] = uintBitsToFloat(v[c] << 16u);
            w[8u * q + 2u * c + 1u] = uintBitsToFloat(v[c] & 0xffff0000u);
        }
    }
}

void glm53_decode_f16(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 64u;
    [[unroll]] for (uint q = 0u; q < 4u; q++) {
        const uvec4 v = U32x4(b + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            const vec2 h = unpackHalf2x16(v[c]);
            w[8u * q + 2u * c] = h.x;
            w[8u * q + 2u * c + 1u] = h.y;
        }
    }
}

void glm53_decode_f32(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 128u;
    [[unroll]] for (uint q = 0u; q < 8u; q++) {
        const vec4 v = F32x4(b + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) w[4u * q + c] = v[c];
    }
}

// The decoder of a compile-time type id.
void glm53_decode(uint ty, uint64_t row, uint t, out float w[32]) {
    if (ty == GLM53_Q8_0) glm53_decode_q8_0(row, t, w);
    else if (ty == GLM53_Q4_K) glm53_decode_q4_k(row, t, w);
#ifdef GLM53_EXT_TYPES
    else if (ty == GLM53_Q4_0) glm53_decode_q4_0(row, t, w);
    else if (ty == GLM53_Q5_K) glm53_decode_q5_k(row, t, w);
    else if (ty == GLM53_Q6_K) glm53_decode_q6_k(row, t, w);
    else if (ty == GLM53_IQ4_NL) glm53_decode_iq4_nl(row, t, w);
    else if (ty == GLM53_IQ2_S) glm53_decode_iq2_s(row, t, w);
#endif
    else if (ty == GLM53_Q2_K) glm53_decode_q2_k(row, t, w);
    else if (ty == GLM53_IQ2_XXS) glm53_decode_iq2_xxs(row, t, w);
    else if (ty == GLM53_BF16) glm53_decode_bf16(row, t, w);
    else if (ty == GLM53_F16) glm53_decode_f16(row, t, w);
    else glm53_decode_f32(row, t, w);
}

// The dot of a decoded task with 32 f32 activations at `x` (16-byte aligned).
float glm53_task_dot(in float w[32], uint64_t x) {
    float acc = 0.0;
    [[unroll]] for (uint q = 0u; q < 8u; q++) {
        const vec4 v = F32x4(x + 16u * q).v[0];
        acc += w[4u * q] * v.x + w[4u * q + 1u] * v.y + w[4u * q + 2u] * v.z + w[4u * q + 3u] * v.w;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Dot products straight from the blocks, for the decode GEMVs of the routed
// experts, which are bound by instructions and loads per weight: the codes
// become exact f16 values through a table, signs are an XOR of the f16 sign
// bits, and a group's scale multiplies its sum once instead of every
// weight. On G17C an f32 fmul widens an f16 source for free but an ffma
// takes two moves for it, so the products are fmuls and the sums separate
// adds (`precise`). The products round in another order than
// glm53_decode + glm53_task_dot; the decoded values are the same.
// ---------------------------------------------------------------------------
// A dynamically indexed table must be a plain global with an initializer
// (the compiler then reads it from the shader's constant data); a `const`
// array is copied to per-thread scratch first.
uvec4 glm53_iq2xxs_grid_h[256] = uvec4[256](
#include "glm53_iq2xxs_grid_h.glsl"
);
uvec4 glm53_iq2xxs_signs_h[128] = uvec4[128](
#include "glm53_iq2xxs_signs_h.glsl"
);

// The block scale of an IQ2_XXS task, 0.125 d (2 s + 1), pinned.
#ifdef LILY_PRECISE_LEGACY
#define GLM53_IQ2_XXS_SCALE(name, d, aux1) \
    precise float name = 0.125 * (d) * float(2u * ((aux1) >> 28u) + 1u)
#else
float glm53_iq2_xxs_scale(float d, uint aux1) {
    precise float k = 0.125 * d * float(2u * (aux1 >> 28u) + 1u);
    return k;
}
#define GLM53_IQ2_XXS_SCALE(name, d, aux1) const float name = glm53_iq2_xxs_scale(d, aux1)
#endif

// IQ2_XXS task t of `row` against the 32 activations xv (as decoded by
// glm53_decode_iq2_xxs, then dotted). Loads index the row as 16-bit words
// (66-byte blocks are 33 words).
float glm53_dot_iq2_xxs(uint64_t row, uint t, in vec4 xv[8]) {
    const uint bw = (t >> 3u) * 33u;
    const uint qw = bw + 1u + 4u * (t & 7u);
    const float d = glm53_h2f(uint(U16(row).v[bw]));
    const uint aux0 = uint(U16(row).v[qw]) | (uint(U16(row).v[qw + 1u]) << 16u);
    const uint aux1 = uint(U16(row).v[qw + 2u]) | (uint(U16(row).v[qw + 3u]) << 16u);
    precise float s = 0.0;
    [[unroll]] for (uint l = 0u; l < 4u; l++) {
        const uint gi = (aux0 >> (8u * l)) & 0xffu;
        const uint si = (aux1 >> (7u * l)) & 127u;
#if defined(GLM53_SIGN_ALU)
        // Signs of values p and p + 4 to bits 15 and 31 of word p: q holds
        // sign k at bits k and 12 + k (the parity bit, sign 7, at 19).
        const uint q = si + (si << 12u) + ((bitCount(si) & 1u) << 19u);
        const uvec4 sg = (uvec4(q) << uvec4(15u, 14u, 13u, 12u)) & 0x80008000u;
#else
        const uvec4 sg = glm53_iq2xxs_signs_h[si];
#endif
        const uvec4 g = glm53_iq2xxs_grid_h[gi] ^ sg;
        const vec4 a = xv[2u * l], b = xv[2u * l + 1u];
        const vec2 p0 = unpackHalf2x16(g.x), p1 = unpackHalf2x16(g.y);
        const vec2 p2 = unpackHalf2x16(g.z), p3 = unpackHalf2x16(g.w);
        precise float e = ((p0.x * a.x + p0.y * b.x) + (p1.x * a.y + p1.y * b.y))
                        + ((p2.x * a.z + p2.y * b.z) + (p3.x * a.w + p3.y * b.w));
        s += e;
    }
    // Scale and product pinned: glm53_dot2_iq2_xxs gives the same.
    GLM53_IQ2_XXS_SCALE(k, d, aux1);
    GLM53_PRECISE_MUL(v, k, s);
    return v;
}

// glm53_dot_iq2_xxs for two activation rows from one decode: each row's sum
// is computed exactly as glm53_dot_iq2_xxs computes it.
vec2 glm53_dot2_iq2_xxs(uint64_t row, uint t, in vec4 xa[8], in vec4 xb[8]) {
    const uint bw = (t >> 3u) * 33u;
    const uint qw = bw + 1u + 4u * (t & 7u);
    const float d = glm53_h2f(uint(U16(row).v[bw]));
    const uint aux0 = uint(U16(row).v[qw]) | (uint(U16(row).v[qw + 1u]) << 16u);
    const uint aux1 = uint(U16(row).v[qw + 2u]) | (uint(U16(row).v[qw + 3u]) << 16u);
    precise float s0 = 0.0, s1 = 0.0;
    [[unroll]] for (uint l = 0u; l < 4u; l++) {
        const uint gi = (aux0 >> (8u * l)) & 0xffu;
        const uint si = (aux1 >> (7u * l)) & 127u;
        const uvec4 g = glm53_iq2xxs_grid_h[gi] ^ glm53_iq2xxs_signs_h[si];
        const vec2 p0 = unpackHalf2x16(g.x), p1 = unpackHalf2x16(g.y);
        const vec2 p2 = unpackHalf2x16(g.z), p3 = unpackHalf2x16(g.w);
        {
            const vec4 a = xa[2u * l], b = xa[2u * l + 1u];
            precise float e = ((p0.x * a.x + p0.y * b.x) + (p1.x * a.y + p1.y * b.y))
                            + ((p2.x * a.z + p2.y * b.z) + (p3.x * a.w + p3.y * b.w));
            s0 += e;
        }
        {
            const vec4 a = xb[2u * l], b = xb[2u * l + 1u];
            precise float e = ((p0.x * a.x + p0.y * b.x) + (p1.x * a.y + p1.y * b.y))
                            + ((p2.x * a.z + p2.y * b.z) + (p3.x * a.w + p3.y * b.w));
            s1 += e;
        }
    }
    GLM53_IQ2_XXS_SCALE(k, d, aux1);
    GLM53_PRECISE_MUL(v0, k, s0);
    GLM53_PRECISE_MUL(v1, k, s1);
    return vec2(v0, v1);
}
