//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};

/// Errors that can occur when attempting to prepare values for the kv cache
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[allow(clippy::module_name_repetitions)]
pub enum KvCacheConversionError {
    /// Sequence id conversion to i32 failed
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    /// Position 0 conversion to i32 failed
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    /// Position 1 conversion to i32 failed
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
}

impl LlamaContext<'_> {
    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, src, dest, 0, size) }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is copied up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is copied starting from `p0`.
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn copy_kv_cache_seq(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, src, dest, p0, p1) };
        Ok(())
    }

    /// Clear the kv cache for the given sequence within the specified range `[p0, p1)`
    /// Returns `false` only when partial sequence removals fail. Full sequence removals always succeed.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful. If the sequence id or
    /// either position exceeds the maximum i32 value, no removal is attempted and an `Err` is returned.
    ///
    /// # Parameters
    /// * `src` - The sequence id to clear the cache for. If `None`, matches all sequences
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from `p0`.
    ///
    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`].
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, KvCacheConversionError> {
        let src = src
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::SeqIdTooLarge)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        Ok(unsafe { llama_cpp_sys_2::llama_memory_seq_rm(mem, src, p0, p1) })
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        // clear both metadata and data buffers to match previous semantics
        unsafe { llama_cpp_sys_2::llama_memory_clear(mem, true) }
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn llama_kv_cache_seq_keep(&mut self, seq_id: i32) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_keep(mem, seq_id) }
    }

    #[allow(clippy::doc_markdown)]
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `delta` - The relative position to add to the tokens
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_add(mem, seq_id, p0, p1, delta) };
        Ok(())
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is `RoPEd`, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `d` - The factor to divide the positions by
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        d: NonZeroU8,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let d = c_int::from(d.get());
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_div(mem, seq_id, p0, p1, d) }
        Ok(())
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_pos_max(mem, seq_id) }
    }

    /// Returns the smallest position present in the KV cache for the specified sequence,
    /// or `-1` if there is no data in the cache for that sequence.
    ///
    /// On a sliding-window-attention (SWA) model this can be greater than zero even
    /// though earlier tokens were once decoded — they've been evicted from the
    /// local layers' KV buffer. Callers performing a prefix-diff rewind must
    /// check `pos_min` against their target rewind position and re-prefill from
    /// scratch if the target is older than what's still cached.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the min position for
    #[must_use]
    pub fn kv_cache_seq_pos_min(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_pos_min(mem, seq_id) }
    }

    /// Per-buffer memory breakdown for this context, in bytes. Each entry
    /// names a buffer type (`"CPU"`, `"Metal"`, `"CUDA0"`, …) and reports
    /// how much memory the model, context (KV cache), and compute buffers
    /// each consume on that device. Wraps the C++-only
    /// `llama_get_memory_breakdown` via a FFI-friendly shim.
    ///
    /// Returns an empty `Vec` if the context has no allocated memory yet.
    /// Logs to debug and returns empty on FFI failure.
    pub fn memory_breakdown(&self) -> Vec<MemoryBreakdownEntry> {
        use std::ffi::CStr;
        let mut entries_ptr: *mut llama_cpp_sys_2::llama_rs_mem_entry =
            std::ptr::null_mut();
        let mut count: usize = 0;
        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_get_memory_breakdown(
                self.context.as_ptr(),
                &mut entries_ptr,
                &mut count,
            )
        };
        if !crate::status_is_ok(rc) || entries_ptr.is_null() || count == 0 {
            // Make sure we still free in the rare case the C side allocated
            // partial output before returning a non-OK status.
            if !entries_ptr.is_null() {
                unsafe {
                    llama_cpp_sys_2::llama_rs_mem_entries_free(entries_ptr, count);
                }
            }
            return Vec::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(entries_ptr, count) };
        let mut out: Vec<MemoryBreakdownEntry> = Vec::with_capacity(count);
        for entry in slice {
            let name = if entry.buft_name.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(entry.buft_name) }
                    .to_string_lossy()
                    .into_owned()
            };
            out.push(MemoryBreakdownEntry {
                buft_name: name,
                model_bytes: entry.model_bytes,
                context_bytes: entry.context_bytes,
                compute_bytes: entry.compute_bytes,
            });
        }
        unsafe {
            llama_cpp_sys_2::llama_rs_mem_entries_free(entries_ptr, count);
        }
        out
    }
}

/// One row of a context's memory breakdown — what one buffer type
/// (CPU / Metal / CUDA0 / …) holds for the model, the KV cache, and
/// compute scratch buffers, in bytes.
#[derive(Debug, Clone)]
pub struct MemoryBreakdownEntry {
    /// Name of the buffer type ("CPU", "Metal", "CUDA0", …).
    pub buft_name: String,
    /// Model weights resident on this buffer.
    pub model_bytes: usize,
    /// Context allocations (KV cache, embedding output, etc.).
    pub context_bytes: usize,
    /// Compute scratch buffers reserved for graph evaluation.
    pub compute_bytes: usize,
}

impl MemoryBreakdownEntry {
    /// Total bytes attributed to this buffer.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.model_bytes + self.context_bytes + self.compute_bytes
    }
}
