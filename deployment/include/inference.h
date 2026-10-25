#pragma once

#include "model.h"

#include <cstdint>
#include <string>
#include <vector>

const Tensor& find_tensor(
    const Model& model,
    const std::string& name
);

std::vector<float> build_embeddings(
    const Model& model,
    const std::vector<uint32_t>& token_ids
);

std::vector<float> layer_norm(
    const std::vector<float>& input,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t sequence_length,
    uint32_t embed_dim
);

std::vector<float> linear(
    const std::vector<float>& input,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t sequence_length,
    uint32_t input_dim,
    uint32_t output_dim
);

std::vector<float> attention(
    const Model& model,
    const std::vector<float>& normalized,
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V
);

std::vector<float> transformer_block_tail(
    const Model& model,
    const std::vector<float>& input,
    const std::vector<float>& attention_output
);

std::vector<float> output_logits(
    const Model& model,
    const std::vector<float>& input
);

std::vector<float> forward(
    const Model& model,
    const std::vector<uint32_t>& token_ids
);