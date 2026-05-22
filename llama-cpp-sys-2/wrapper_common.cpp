#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <stdint.h>

#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result) {
    if (!result) {
        return;
    }
    if (result->prompt) {
        std::free(result->prompt);
    }
    if (result->grammar) {
        std::free(result->grammar);
    }
    if (result->parser) {
        std::free(result->parser);
    }
    if (result->generation_prompt) {
        std::free(result->generation_prompt);
    }
    if (result->grammar_triggers) {
        for (size_t i = 0; i < result->grammar_triggers_count; ++i) {
            std::free(result->grammar_triggers[i].value);
        }
        std::free(result->grammar_triggers);
    }
    if (result->preserved_tokens) {
        for (size_t i = 0; i < result->preserved_tokens_count; ++i) {
            std::free(result->preserved_tokens[i]);
        }
        std::free(result->preserved_tokens);
    }
    if (result->additional_stops) {
        for (size_t i = 0; i < result->additional_stops_count; ++i) {
            std::free(result->additional_stops[i]);
        }
        std::free(result->additional_stops);
    }
    result->prompt = nullptr;
    result->grammar = nullptr;
    result->parser = nullptr;
    result->generation_prompt = nullptr;
    result->chat_format = 0;
    result->grammar_lazy = false;
    result->grammar_triggers = nullptr;
    result->grammar_triggers_count = 0;
    result->preserved_tokens = nullptr;
    result->preserved_tokens_count = 0;
    result->additional_stops = nullptr;
    result->additional_stops_count = 0;
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// -----------------------------------------------------------------------------
// Memory breakdown
// -----------------------------------------------------------------------------
//
// `llama_get_memory_breakdown` lives in src/llama-ext.h (WIP staging
// header). It returns a std::map keyed on opaque buffer-type handles, which
// bindgen can't reflect into Rust. We flatten the map here into an array of
// {name, model, context, compute} records that crosses the FFI boundary
// cleanly. Buffer-type names come from ggml_backend_buft_name() ("CPU",
// "Metal", "CUDA0", etc.).

#include "llama.cpp/src/llama-ext.h"
#include "llama.cpp/ggml/include/ggml-backend.h"

extern "C" llama_rs_status llama_rs_get_memory_breakdown(
    const struct llama_context * ctx,
    struct llama_rs_mem_entry ** out_entries,
    size_t * out_count) {
    if (!ctx || !out_entries || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_entries = nullptr;
    *out_count = 0;

    try {
        const auto breakdown = llama_get_memory_breakdown(ctx);
        const size_t n = breakdown.size();
        if (n == 0) {
            return LLAMA_RS_STATUS_OK;
        }

        auto * arr = static_cast<llama_rs_mem_entry *>(
            std::calloc(n, sizeof(llama_rs_mem_entry)));
        if (!arr) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }

        size_t i = 0;
        for (const auto & kv : breakdown) {
            const char * raw_name = kv.first ? ggml_backend_buft_name(kv.first) : "?";
            arr[i].buft_name = llama_rs_dup_string(raw_name ? raw_name : "?");
            if (!arr[i].buft_name) {
                // free what we managed so far
                for (size_t j = 0; j < i; ++j) {
                    std::free(arr[j].buft_name);
                }
                std::free(arr);
                return LLAMA_RS_STATUS_ALLOCATION_FAILED;
            }
            arr[i].model_bytes = kv.second.model;
            arr[i].context_bytes = kv.second.context;
            arr[i].compute_bytes = kv.second.compute;
            ++i;
        }

        *out_entries = arr;
        *out_count = n;
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_mem_entries_free(
    struct llama_rs_mem_entry * entries,
    size_t count) {
    if (!entries) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(entries[i].buft_name);
    }
    std::free(entries);
}
