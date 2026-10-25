#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "generator.h"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <string>

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
}

size_t get_current_rss_kb()
{
    std::ifstream status_file("/proc/self/status");

    if (!status_file)
        return 0;

    std::string line;

    while (std::getline(status_file, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
        {
            size_t value_start = line.find_first_of("0123456789");

            if (value_start == std::string::npos)
                return 0;

            return std::stoul(line.substr(value_start));
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: yatt <model> [prompt]\n";
        return 1;
    }

    const std::string model_path = argv[1];

    std::string prompt = "First Citizen:";

    if (argc >= 3)
        prompt = argv[2];

    using Clock = std::chrono::steady_clock;

    // --------------------------------------------------------
    // Load model
    // --------------------------------------------------------

    Model model;

    auto load_start = Clock::now();

    if (!load_model(model_path, model))
        return 1;

    auto load_end = Clock::now();

    // --------------------------------------------------------
    // Tokenizer
    // --------------------------------------------------------

    Tokenizer tokenizer(model);

    auto tokenize_start = Clock::now();

    std::vector<uint32_t> prompt_tokens =
        tokenizer.encode(prompt);

    auto tokenize_end = Clock::now();

    // --------------------------------------------------------
    // Generation
    // --------------------------------------------------------

    constexpr uint32_t max_new_tokens = 100;
    constexpr float temperature = 0.8f;

    auto generation_start = Clock::now();

    std::vector<uint32_t> generated =
        generate(
            model,
            prompt_tokens,
            max_new_tokens,
            temperature
        );

    auto generation_end = Clock::now();
    
    const size_t rss_kb = get_current_rss_kb();

    // --------------------------------------------------------
    // Decode
    // --------------------------------------------------------

    std::string output =
        tokenizer.decode(generated);

    // --------------------------------------------------------
    // Timing
    // --------------------------------------------------------

    const double load_ms =
        std::chrono::duration<double, std::milli>(
            load_end - load_start
        ).count();

    const double tokenize_ms =
        std::chrono::duration<double, std::milli>(
            tokenize_end - tokenize_start
        ).count();

    const double generation_ms =
        std::chrono::duration<double, std::milli>(
            generation_end - generation_start
        ).count();

    const uint32_t generated_tokens =
        static_cast<uint32_t>(
            generated.size() - prompt_tokens.size()
        );

    const double total_ms =
        load_ms + tokenize_ms + generation_ms;

    const double ms_per_token =
        generated_tokens > 0
            ? generation_ms / generated_tokens
            : 0.0;

    const double tokens_per_second =
        generation_ms > 0.0
            ? generated_tokens / (generation_ms / 1000.0)
            : 0.0;

    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "YATT Baseline Benchmark\n";
    std::cout << "========================================\n";

    std::cout << "Model:              " << model_path << "\n";
    std::cout << "Prompt:             " << prompt << "\n";
    std::cout << "Prompt tokens:      " << prompt_tokens.size() << "\n";
    std::cout << "Generated tokens:   " << generated_tokens << "\n";
    std::cout << "Context length:     " << model.context_length << "\n";
    std::cout << "Embed dimension:    " << model.embed_dim << "\n";
    std::cout << "FFN dimension:      " << model.ffn_dim << "\n";
    std::cout << "Attention heads:    " << model.num_heads << "\n";

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nTiming:\n";
    std::cout << "  Model load:       " << load_ms << " ms\n";
    std::cout << "  Tokenization:     " << tokenize_ms << " ms\n";
    std::cout << "  Generation:       " << generation_ms << " ms\n";
    std::cout << "  Total:            " << total_ms << " ms\n";

    std::cout << "\nGeneration:\n";
    std::cout << "  Avg latency:      " << ms_per_token << " ms/token\n";
    std::cout << "  Throughput:       " << tokens_per_second << " tokens/sec\n";

    std::cout << "\nMemory:\n";
    std::cout << "  Process RSS:      "
            << rss_kb / 1024.0
            << " MB\n";

    std::cout << "\nOutput:\n";
    std::cout << output << "\n";

    std::cout << "========================================\n";

    return 0;
}