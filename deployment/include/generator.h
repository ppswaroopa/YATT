#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

std::vector<uint32_t> generate(
    const Model& model,
    const std::vector<uint32_t>& prompt,
    uint32_t max_new_tokens,
    float temperature
);