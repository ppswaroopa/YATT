#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Tensor
{
    std::string name;
    std::vector<uint32_t> shape;
    uint32_t data_type;
    uint64_t data_bytes;
    std::vector<float> data;
};

struct Model
{
    uint32_t version;
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t embed_dim;
    uint32_t ffn_dim;
    uint32_t num_heads;

    std::vector<char> vocabulary;

    std::vector<Tensor> tensors;
};

bool load_model(const std::string& path, Model& model);