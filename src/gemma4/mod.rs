//! Gemma 4 26B-A4B (`gemma4` GGUF): configuration and the model graph.
//! The graph follows llama.cpp's (MIT, `src/models/gemma4.cpp`); lily loads
//! the published GGUF files as they are.

pub mod config;
pub mod model;

pub use config::Config;
pub use model::{Gemma4Model, Gemma4State, Logits, Scratch};
