//! Speculative decoding via the `common_speculative` framework.
//!
//! Supports both the Gemma-4 "assistant" MTP drafter (a tiny head that shares
//! the backbone's K/V) and the generic separate-draft-model path. The drafter
//! type is auto-selected from the draft model's `general.architecture`, so a
//! non-Gemma draft model transparently uses the standard `draft-simple` path.
//!
//! The verify/accept loop itself stays with the caller: decode
//! `[id_last, draft...]` on the target, then sample each row through the normal
//! sampler chain and accept a draft iff the target's own sample matches it
//! (lossless and distribution-preserving even at temperature > 0, and grammar
//! still applies). These bindings only own the drafter state machine.

use crate::context::LlamaContext;
use crate::llama_batch::LlamaBatch;
use crate::token::LlamaToken;
use std::os::raw::c_int;

/// A speculative-decoding drafter bound to a target (backbone) context and a
/// draft context.
///
/// Lifetime: the underlying `common_speculative` keeps raw pointers to both
/// contexts but does not borrow them (so all three can be stored together).
/// The caller must keep both contexts alive for at least as long as this value.
pub struct LlamaSpeculative {
    spec: *mut llama_cpp_sys_2::common_speculative,
}

impl LlamaSpeculative {
    /// Initialise speculative decoding for `ctx_tgt` (backbone) drafted by
    /// `ctx_dft`. `n_max` is the maximum draft length per step. Returns `None`
    /// if the framework could not initialise the drafter.
    #[must_use]
    pub fn new(ctx_tgt: &LlamaContext, ctx_dft: &LlamaContext, n_max: i32) -> Option<Self> {
        let spec = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_init(
                ctx_tgt.context.as_ptr(),
                ctx_dft.context.as_ptr(),
                n_max,
                1,
            )
        };
        if spec.is_null() {
            None
        } else {
            Some(Self { spec })
        }
    }

    /// Seed the drafter with the prompt token history (call once after `new`).
    pub fn begin(&self, seq_id: i32, prompt: &[LlamaToken]) {
        unsafe {
            llama_cpp_sys_2::llama_rs_speculative_begin(
                self.spec,
                seq_id,
                prompt.as_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                prompt.len(),
            );
        }
    }

    /// Feed a just-decoded target batch so the drafter can capture backbone
    /// state (hidden state + shared K/V). Call right after decoding `batch` on
    /// the target context.
    pub fn process(&self, batch: &LlamaBatch) -> bool {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_process(self.spec, batch.llama_batch) }
    }

    /// Generate draft tokens continuing from `id_last` at position `n_past`,
    /// replacing the contents of `out`. `prompt` is the full token history so
    /// far. At most `max_draft` tokens are produced.
    pub fn draft(
        &self,
        seq_id: i32,
        id_last: LlamaToken,
        n_past: i32,
        prompt: &[LlamaToken],
        max_draft: usize,
        out: &mut Vec<LlamaToken>,
    ) {
        out.clear();
        out.resize(max_draft, LlamaToken(0));
        let n = unsafe {
            llama_cpp_sys_2::llama_rs_speculative_draft(
                self.spec,
                seq_id,
                id_last.0,
                n_past,
                prompt.as_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                prompt.len(),
                out.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                max_draft as c_int,
            )
        };
        out.truncate(if n > 0 { n as usize } else { 0 });
    }

    /// Notify the drafter how many of its last drafts were accepted (rolls its
    /// internal hidden-state seed back to the accepted position).
    pub fn accept(&self, seq_id: i32, n_accepted: u16) {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_accept(self.spec, seq_id, n_accepted) }
    }
}

impl Drop for LlamaSpeculative {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_speculative_free(self.spec) }
    }
}
