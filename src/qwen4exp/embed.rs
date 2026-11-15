//! The token embedding read from the file instead of mapped for the GPU:
//! Qwen3.8's Q2 GGUF under a memory limit (`--memory-limit-gb`) leaves its
//! 1.27 GB table off the GPU this way. The trunk's rows are copied by the
//! host for the tokens it feeds, which it knows before each pass reads them
//! (a prompt chunk, a verify pass, a decode step), so the trunk embeds
//! exactly what the mapped table held. The draft head embeds the tokens the
//! host staged from those rows too (a prompt chunk's, a verify pass's
//! inputs and accepted drafts). Tokens the GPU draws within a draft pass
//! (the step's new draw, the head's own proposals) come from a resident copy
//! of the rows of its narrow LM head (`spec.rs`; 505 MB, so the net saving
//! with the head is 767 MB), where another id embeds as zeros, which can
//! only cost acceptance; the next draft pass rewrites such a new draw's row
//! in the head's cache once it is the staged pending token.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use crate::tensor::{DType, Tensor};

/// A BF16 embedding table in a file: `rows` rows of `cols` values from
/// byte `offset`.
pub struct PagedEmbed {
    file: File,
    offset: u64,
    rows: usize,
    cols: usize,
}

impl PagedEmbed {
    pub fn open(path: &Path, offset: u64, rows: usize, cols: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let end = offset + (rows * cols * 2) as u64;
        ensure!(
            file.metadata()?.len() >= end,
            "{} ends before its embedding table",
            path.display()
        );
        Ok(Self { file, offset, rows, cols })
    }

    fn row_bytes(&self) -> usize {
        self.cols * 2
    }

    /// The bytes of rows `start..start + n`.
    pub fn read_rows(&self, start: usize, n: usize) -> Result<Vec<u8>> {
        ensure!(start + n <= self.rows, "rows {start}..{} past {}", start + n, self.rows);
        let mut b = vec![0u8; n * self.row_bytes()];
        self.file
            .read_exact_at(&mut b, self.offset + (start * self.row_bytes()) as u64)?;
        Ok(b)
    }

    /// Writes the rows of `tokens` into `out` (bfloat `[tokens.len(),
    /// cols]` at its start). The GPU must not be reading `out`.
    pub fn fill(&self, tokens: &[u32], out: &Tensor) -> Result<()> {
        ensure!(
            out.dtype() == DType::BF16 && out.numel() >= tokens.len() * self.cols,
            "embedding rows into {:?} of {}",
            out.dtype(),
            out.numel()
        );
        let rb = self.row_bytes();
        let mut b = vec![0u8; tokens.len() * rb];
        for (row, &t) in b.chunks_exact_mut(rb).zip(tokens) {
            let t = t as usize;
            ensure!(t < self.rows, "token id {t} outside the table of {}", self.rows);
            self.file.read_exact_at(row, self.offset + (t * rb) as u64)?;
        }
        out.view(0, &[tokens.len() * self.cols])?.write_bytes(&b)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/embed.rs"]
mod tests;
