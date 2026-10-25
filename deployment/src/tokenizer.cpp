#include "tokenizer.h"

#include <stdexcept>

Tokenizer::Tokenizer(const Model& model)
    : vocabulary(model.vocabulary),
      token_lookup{}
{
    token_lookup.fill(-1);

    for (uint32_t token_id = 0;
         token_id < vocabulary.size();
         ++token_id)
    {
        unsigned char c =
            static_cast<unsigned char>(
                vocabulary[token_id]
            );

        if (token_lookup[c] != -1)
        {
            throw std::runtime_error(
                "Duplicate character in vocabulary"
            );
        }

        token_lookup[c] =
            static_cast<int32_t>(token_id);
    }
}

uint32_t Tokenizer::encode_char(char c) const
{
    unsigned char index =
        static_cast<unsigned char>(c);

    int32_t token_id =
        token_lookup[index];

    if (token_id < 0)
    {
        throw std::runtime_error(
            "Character not present in vocabulary"
        );
    }

    return static_cast<uint32_t>(token_id);
}

std::vector<uint32_t> Tokenizer::encode(
    const std::string& text) const
{
    std::vector<uint32_t> tokens;
    tokens.reserve(text.size());

    for (char c : text)
        tokens.push_back(encode_char(c));

    return tokens;
}

char Tokenizer::decode_token(
    uint32_t token_id) const
{
    if (token_id >= vocabulary.size())
    {
        throw std::runtime_error(
            "Token ID out of vocabulary range"
        );
    }

    return vocabulary[token_id];
}

std::string Tokenizer::decode(
    const std::vector<uint32_t>& token_ids) const
{
    std::string text;
    text.reserve(token_ids.size());

    for (uint32_t token_id : token_ids)
        text.push_back(decode_token(token_id));

    return text;
}