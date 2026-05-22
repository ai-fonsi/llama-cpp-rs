#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_vocab;

struct llama_rs_grammar_trigger {
    int type;
    char * value;
    llama_token token;
};

struct llama_rs_chat_template_result {
    char * prompt;
    char * grammar;
    char * parser;
    char * generation_prompt;
    int chat_format;
    bool grammar_lazy;
    struct llama_rs_grammar_trigger * grammar_triggers;
    size_t grammar_triggers_count;
    char ** preserved_tokens;
    size_t preserved_tokens_count;
    char ** additional_stops;
    size_t additional_stops_count;
};

#include "wrapper_utils.h"

// Per-buffer memory breakdown for a context, flattened into a C struct so
// it can cross the FFI boundary. The C++ original lives in
// `src/llama-ext.h` and returns a `std::map<ggml_backend_buffer_type_t, ...>`
// which bindgen can't represent. `llama_rs_get_memory_breakdown` walks the
// map and copies each entry into a flat array of these.
struct llama_rs_mem_entry {
    char * buft_name;       // owned; freed by llama_rs_mem_entries_free
    size_t model_bytes;     // bytes allocated for model weights on this buft
    size_t context_bytes;   // bytes allocated for context (KV cache etc.)
    size_t compute_bytes;   // bytes allocated for compute scratch buffers
};

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result);
void llama_rs_string_free(char * ptr);

// Allocates an array of `*out_count` `llama_rs_mem_entry` records describing
// per-buffer memory use for `ctx`, in the same shape llama.cpp's own
// `llama_get_memory_breakdown_print` walks. Call `llama_rs_mem_entries_free`
// to release.
llama_rs_status llama_rs_get_memory_breakdown(
    const struct llama_context * ctx,
    struct llama_rs_mem_entry ** out_entries,
    size_t * out_count);

void llama_rs_mem_entries_free(
    struct llama_rs_mem_entry * entries,
    size_t count);

#ifdef __cplusplus
}
#endif
