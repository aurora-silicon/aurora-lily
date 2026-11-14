//! Qwen3.6-35B-A3B (`qwen35moe` GGUF): configuration and the model graph.
//! The graph and the file's layout are llama.cpp's (MIT): its converter
//! wrote the GGUFs lily loads as they are, and `src/models/qwen35moe.cpp`
//! defines what each tensor means.

pub mod config;
pub mod model;
pub mod schema;

pub use config::Config;
pub use model::{Logits, Qwen36Model, Qwen36State, Scratch};
