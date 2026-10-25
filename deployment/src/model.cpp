#include "model.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{

uint32_t read_u32(std::ifstream& file)
{
    uint32_t value;

    file.read(
        reinterpret_cast<char*>(&value),
        sizeof(value)
    );

    if (!file)
    {
        throw std::runtime_error("Failed to read uint32");
    }

    return value;
}


uint64_t read_u64(std::ifstream& file)
{
    uint64_t value;

    file.read(
        reinterpret_cast<char*>(&value),
        sizeof(value)
    );

    if (!file)
    {
        throw std::runtime_error("Failed to read uint64");
    }

    return value;
}


std::string read_string(std::ifstream& file)
{
    uint32_t length = read_u32(file);

    std::string value(length, '\0');

    file.read(
        value.data(),
        length
    );

    if (!file)
    {
        throw std::runtime_error("Failed to read string");
    }

    return value;
}

}


bool load_model(const std::string& path, Model& model)
{
    std::ifstream file(
        path,
        std::ios::binary
    );

    if (!file)
    {
        std::cerr << "Failed to open model: "
                  << path << "\n";

        return false;
    }

    // --------------------------------------------------
    // Magic
    // --------------------------------------------------

    char magic[4];

    file.read(magic, 4);

    if (!file)
    {
        std::cerr << "Failed to read model magic\n";
        return false;
    }

    if (magic[0] != 'Y' ||
        magic[1] != 'A' ||
        magic[2] != 'T' ||
        magic[3] != 'T')
    {
        std::cerr << "Invalid YATT model\n";
        return false;
    }

    // --------------------------------------------------
    // Header
    // --------------------------------------------------

    model.version = read_u32(file);
    model.vocab_size = read_u32(file);
    model.context_length = read_u32(file);
    model.embed_dim = read_u32(file);
    model.ffn_dim = read_u32(file);
    model.num_heads = read_u32(file);

    uint32_t vocab_count = read_u32(file);

    if (vocab_count != model.vocab_size)
    {
        throw std::runtime_error(
            "Vocabulary size does not match model metadata"
        );
    }

    model.vocabulary.resize(vocab_count);

    for (uint32_t i = 0; i < vocab_count; ++i)
    {
        file.read(
            &model.vocabulary[i],
            sizeof(char)
        );

        if (!file)
        {
            throw std::runtime_error(
                "Failed to read vocabulary"
            );
        }
    }

    uint32_t tensor_count = read_u32(file);

    // --------------------------------------------------
    // Tensor metadata
    // --------------------------------------------------

    model.tensors.reserve(tensor_count);

    for (uint32_t i = 0; i < tensor_count; ++i)
    {
        Tensor tensor;

        tensor.name = read_string(file);

        uint32_t ndim = read_u32(file);

        tensor.shape.resize(ndim);

        for (uint32_t d = 0; d < ndim; ++d)
        {
            tensor.shape[d] = read_u32(file);
        }

        tensor.data_type = read_u32(file);
        tensor.data_bytes = read_u64(file);

        model.tensors.push_back(std::move(tensor));
    }

    // --------------------------------------------------
    // Tensor data
    // --------------------------------------------------

    for (Tensor& tensor : model.tensors)
    {
        if (tensor.data_type != 1)
        {
            throw std::runtime_error(
                "Unsupported tensor data type"
            );
        }

        if (tensor.data_bytes % sizeof(float) != 0)
        {
            throw std::runtime_error(
                "Tensor byte size is not divisible by float size"
            );
        }

        size_t element_count =
            tensor.data_bytes / sizeof(float);

        tensor.data.resize(element_count);

        file.read(
            reinterpret_cast<char*>(tensor.data.data()),
            tensor.data_bytes
        );

        if (!file)
        {
            throw std::runtime_error(
                "Failed to read tensor data: " +
                tensor.name
            );
        }
    }

    return true;
}