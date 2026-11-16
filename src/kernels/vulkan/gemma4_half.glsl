// Gemma 4's f16 activations (the routed experts' f16 inputs and mid in
// gemma4.comp): f32 to f16 bits, clamped to +-65504 first, so finite
// overflow and both infinities saturate to the largest finite half (a NaN
// stays a NaN: clamp would make it -65504). The conversion is a plain
// OpFConvert, without a rounding-mode decoration: on this driver it rounds
// to nearest even, where packHalf2x16 truncates (the boundary cases are
// pinned by `f16_conversion_boundaries` in tests/unit/kernels/gemma4.rs).
uint g4_half(float v) {
    return uint(float16BitsToUint16(float16_t(isnan(v) ? v : clamp(v, -65504.0, 65504.0))));
}
