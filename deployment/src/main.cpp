#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "generator.h"

#include <iostream>


void print_model(const Model& model)
{
    std::cout << "YATT model\n";
    std::cout << "----------\n";

    std::cout << "version:        "
              << model.version << "\n";

    std::cout << "vocab size:     "
              << model.vocab_size << "\n";

    std::cout << "context length: "
              << model.context_length << "\n";

    std::cout << "embed dim:      "
              << model.embed_dim << "\n";

    std::cout << "FFN dim:        "
              << model.ffn_dim << "\n";

    std::cout << "heads:          "
              << model.num_heads << "\n";

    std::cout << "tensors:        "
              << model.tensors.size() << "\n\n";


    for (const Tensor& tensor : model.tensors)
    {
        std::cout << tensor.name << " ";

        std::cout << "[";

        for (size_t i = 0;
             i < tensor.shape.size();
             ++i)
        {
            std::cout << tensor.shape[i];

            if (i + 1 < tensor.shape.size())
            {
                std::cout << ",";
            }
        }

        std::cout << "]";

        std::cout << " "
                  << tensor.data_bytes
                  << " bytes\n";
    }

    std::vector<uint32_t> token_ids = {
    18, 47, 56, 57, 58, 1, 15, 47,
    58, 47, 64, 43, 52, 10
    };

    std::vector<float> embeddings =
        build_embeddings(model, token_ids);

    const Tensor& ln1_weight =
    find_tensor(model, "ln1.weight");

    const Tensor& ln1_bias =
        find_tensor(model, "ln1.bias");

    std::vector<float> normalized =
        layer_norm(
            embeddings,
            ln1_weight,
            ln1_bias,
            token_ids.size(),
            model.embed_dim
        );

    std::cout << "\nLayerNorm output:\n";

    for (size_t t = 0; t < token_ids.size(); ++t)
    {
        std::cout << "token " << token_ids[t] << ": ";

        for (size_t d = 0; d < 5; ++d)
        {
            std::cout << normalized[t * model.embed_dim + d];

            if (d + 1 < 5)
                std::cout << ", ";
        }

        std::cout << "\n";
    }
    std::cout << "\nEmbedding output:\n";

    for (size_t t = 0; t < token_ids.size(); ++t)
    {
        std::cout << "token " << token_ids[t] << ": ";

        for (size_t d = 0; d < 5; ++d)
        {
            std::cout << embeddings[t * model.embed_dim + d];

            if (d + 1 < 5)
                std::cout << ", ";
        }

        std::cout << "\n";
    }

    const Tensor& q_weight =
        find_tensor(model, "q_proj.weight");

    const Tensor& q_bias =
        find_tensor(model, "q_proj.bias");

    const Tensor& k_weight =
        find_tensor(model, "k_proj.weight");

    const Tensor& k_bias =
        find_tensor(model, "k_proj.bias");

    const Tensor& v_weight =
        find_tensor(model, "v_proj.weight");

    const Tensor& v_bias =
        find_tensor(model, "v_proj.bias");

    std::vector<float> Q =
        linear(
            normalized,
            q_weight,
            q_bias,
            token_ids.size(),
            model.embed_dim,
            model.embed_dim
        );

    std::vector<float> K =
        linear(
            normalized,
            k_weight,
            k_bias,
            token_ids.size(),
            model.embed_dim,
            model.embed_dim
        );

    std::vector<float> V =
        linear(
            normalized,
            v_weight,
            v_bias,
            token_ids.size(),
            model.embed_dim,
            model.embed_dim
        );
    std::cout << "\nQ output:\n";
    
    std::vector<float> attention_output =
    attention(
        model,
        normalized,
        Q,
        K,
        V
    );

    std::vector<float> block_output =
    transformer_block_tail(
        model,
        embeddings,
        attention_output
    );

    std::vector<float> logits =
        output_logits(
            model,
            block_output
        );

    for (size_t d = 0; d < 5; ++d)
        std::cout << Q[d] << " ";
    std::cout << "\n";

    std::cout << "\nK output:\n";

    for (size_t d = 0; d < 5; ++d)
        std::cout << K[d] << " ";
    std::cout << "\n";

    std::cout << "\nV output:\n";

    for (size_t d = 0; d < 5; ++d)
        std::cout << V[d] << " ";
    std::cout << "\n";

    std::cout << "\nAttention output:\n";

    for (size_t t = 0; t < token_ids.size(); ++t)
    {
        std::cout << "token " << token_ids[t] << ": ";

        for (size_t d = 0; d < 5; ++d)
        {
            std::cout
                << attention_output[
                    t * model.embed_dim + d
                ];

            if (d + 1 < 5)
                std::cout << ", ";
        }

        std::cout << "\n";
    }

    std::cout << "\nFinal logits:\n";

    size_t last_token_offset =
        (token_ids.size() - 1) * model.vocab_size;

    for (size_t i = 0; i < model.vocab_size; ++i)
    {
        std::cout
            << i
            << ": "
            << logits[last_token_offset + i]
            << "\n";
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr
            << "Usage: yatt <model> [prompt]\n";

        return 1;
    }

    const std::string model_path = argv[1];

    std::string prompt = "First Citizen:";

    if (argc >= 3)
        prompt = argv[2];

    Model model;

    if (!load_model(model_path, model))
        return 1;

    Tokenizer tokenizer(model);

    std::vector<uint32_t> prompt_tokens =
        tokenizer.encode(prompt);

    std::vector<uint32_t> generated =
        generate(
            model,
            prompt_tokens,
            100,
            0.8f
        );

    std::string output =
        tokenizer.decode(generated);

    std::cout << output << "\n";

    return 0;
}