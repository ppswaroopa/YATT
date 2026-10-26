#include "generator.h"
#include "inference.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{

uint32_t sample_token(
    const std::vector<float>& logits,
    float temperature)
{
    if (logits.empty())
        throw std::runtime_error("Cannot sample from empty logits");

    if (temperature <= 0.0f)
    {
        uint32_t best_token = 0;

        for (uint32_t i = 1; i < logits.size(); ++i)
        {
            if (logits[i] > logits[best_token])
                best_token = i;
        }

        return best_token;
    }

    float max_logit = logits[0];

    for (float value : logits)
        max_logit = std::max(max_logit, value);

    std::vector<float> probabilities(logits.size());

    float probability_sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i)
    {
        float value = std::exp(
            (logits[i] - max_logit) / temperature
        );

        probabilities[i] = value;
        probability_sum += value;
    }

    for (float& probability : probabilities)
        probability /= probability_sum;

    static std::random_device rd;
    static std::mt19937 generator(rd());

    std::discrete_distribution<uint32_t> distribution(
        probabilities.begin(),
        probabilities.end()
    );

    return distribution(generator);
}

} // namespace

std::vector<uint32_t> generate(
    const Model& model,
    const std::vector<uint32_t>& prompt,
    uint32_t max_new_tokens,
    float temperature)
{
    if (prompt.empty())
        throw std::runtime_error("Prompt cannot be empty");

    if (prompt.size() > model.context_length)
        throw std::runtime_error(
            "Prompt exceeds model context length");

    std::vector<uint32_t> tokens = prompt;

    tokens.reserve(prompt.size() + max_new_tokens);

    // ------------------------------------------------------------
    // Initialize KV cache
    // ------------------------------------------------------------

    KVCache cache;

    init_kv_cache(
        cache,
        model.context_length,
        model.num_heads,
        model.embed_dim / model.num_heads);

    // ------------------------------------------------------------
    // Prefill cache with prompt
    // ------------------------------------------------------------

    prefill_kv_cache(
        model,
        prompt,
        cache);

    // ------------------------------------------------------------
    // Get logits for the final prompt token.
    //
    // We intentionally use the existing full forward() here.
    // This gives us an exact reference for the first generated
    // token while the KV cache is being initialized.
    // ------------------------------------------------------------

    std::vector<float> logits =
        forward(model, prompt);

    size_t last_offset =
        (prompt.size() - 1) * model.vocab_size;

    std::vector<float> last_logits(
        logits.begin() + last_offset,
        logits.begin() + last_offset + model.vocab_size);

    uint32_t next_token =
        sample_token(
            last_logits,
            temperature);

    tokens.push_back(next_token);

    // ------------------------------------------------------------
    // Decode remaining tokens using KV cache
    // ------------------------------------------------------------

    for (uint32_t step = 1;
         step < max_new_tokens;
         ++step)
    {
        // If the cache is full, rebuild it from the current
        // sliding context window.
        if (cache.sequence_length >= model.context_length)
        {
            const size_t context_start =
                tokens.size() - model.context_length;

            std::vector<uint32_t> context(
                tokens.begin() + context_start,
                tokens.end());

            prefill_kv_cache(
                model,
                context,
                cache);
        }

        // Run exactly one new token through the model.
        logits =
            forward_cached(
                model,
                cache,
                next_token);

        // forward_cached() returns logits for the new token.
        last_logits = std::move(logits);

        next_token =
            sample_token(
                last_logits,
                temperature);

        tokens.push_back(next_token);
    }

    return tokens;
}