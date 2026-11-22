// Gemma 4's int8 activations (gemma4_q8x.comp and the producers'
// `_q8x` variants in gemma4.comp): one rule for every quantizer. A 32-value
// block's scale is max |x| / 127 over its values; a block holding a NaN or
// an infinity gets a NaN scale (its products come out NaN, as the f32
// path's do) and zero codes. Codes are round-to-even(x / scale) clamped to
// [-127, 127] (a subnormal maximum can give 128), converted from a finite
// float; the stored sum is the clamped codes'.

// glslang rejects builtin calls in global const initializers, so the
// quiet-NaN payload is stored as bits and converted at the use site.
const uint G4_Q8X_NAN_BITS = 0x7fc00000u;

bool g4_q8x_nonfinite(float v) {
    return isnan(v) || isinf(v);
}

// |v|, or 0 for a NaN or an infinity (the block is flagged apart).
float g4_q8x_abs(float v) {
    return g4_q8x_nonfinite(v) ? 0.0 : abs(v);
}

// The block's scale from its finite maximum and whether it holds a NaN or
// an infinity.
float g4_q8x_scale(float amax, bool nonfinite) {
    return nonfinite ? uintBitsToFloat(G4_Q8X_NAN_BITS) : amax / 127.0;
}

int g4_q8x_code(float v, float d) {
    return int(clamp(roundEven(d > 0.0 ? v / d : 0.0), -127.0, 127.0));
}

ivec4 g4_q8x_code4(vec4 v, float d) {
    return ivec4(clamp(roundEven(d > 0.0 ? v / d : vec4(0.0)), -127.0, 127.0));
}
