//! Inkling 2 Mini (`inkling2_loop` / `inkling2`): a 26.4B-parameter looped
//! Mixture-of-Experts multimodal model by Thinking Machines Lab.
//!
//! 24 physical layers (prelude 0..6, looped core 6..18 run 3 passes, coda
//! 18..24 = 48 logical layer visits), 128 routed experts top-8 with SITU
//! activation, relative-position attention with zero sink (no RoPE), causal
//! short convolutions (with dynamic per-head taps on global layers), block
//! attention residuals, and rank-64 cross-pass loop controllers.
//!
//! Loads published GGUFs from both `ggml-org/Inkling-2-Mini-GGUF` (BF16, Q8_0,
//! IQ4_NL) and `unsloth/Inkling-2-Mini-GGUF` (including 16 GB UD-Q2_K_XL /
//! UD-IQ2_XXS, DSpark speculative drafter, and `inkling2v` vision mmproj).

pub mod config;
pub mod dspark;
pub mod engine;
pub mod model;
pub mod reference;
pub mod schema;
pub mod tokenizer;
pub mod vision;

pub use config::Config;
pub use engine::{Inkling2Engine, Inkling2Snapshot};
pub use model::{Inkling2Model, Inkling2State};
