#pragma once

#include "model.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class Tokenizer
{
public:
    explicit Tokenizer(const Model& model);

    uint32_t encode_char(char c) const;

    std::vector<uint32_t> encode(
        const std::string& text
    ) const;

    char decode_token(uint32_t token_id) const;

    std::string decode(
        const std::vector<uint32_t>& token_ids
    ) const;

private:
    std::vector<char> vocabulary;
    std::array<int32_t, 256> token_lookup;
};