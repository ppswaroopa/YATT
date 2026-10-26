#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include "inference.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>

const Tensor& find_tensor(const Model& model, const std::string& name)
{
    for (const Tensor& tensor : model.tensors)
    {
        if (tensor.name == name)
            return tensor;
    }

    throw std::runtime_error("Tensor not found: " + name);
}

std::vector<float> build_embeddings(
    const Model& model,
    const std::vector<uint32_t>& token_ids)
{
    const Tensor& token_embedding =
        find_tensor(model, "token_embedding.weight");

    const Tensor& position_embedding =
        find_tensor(model, "position_embedding.weight");

    const uint32_t T = static_cast<uint32_t>(token_ids.size());
    const uint32_t C = model.embed_dim;

    if (T > model.context_length)
        throw std::runtime_error("Input exceeds context length");

    std::vector<float> output(T * C);

    for (uint32_t t = 0; t < T; ++t)
    {
        uint32_t token_id = token_ids[t];

        if (token_id >= model.vocab_size)
            throw std::runtime_error("Token ID out of range");

        for (uint32_t d = 0; d < C; ++d)
        {
            float token_value =
                token_embedding.data[token_id * C + d];

            float position_value =
                position_embedding.data[t * C + d];

            output[t * C + d] =
                token_value + position_value;
        }
    }

    return output;
}

std::vector<float> layer_norm(
    const std::vector<float>& input,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t sequence_length,
    uint32_t embed_dim)
{
    constexpr float epsilon = 1e-5f;

    std::vector<float> output(input.size());

    for (uint32_t t = 0; t < sequence_length; ++t)
    {
        const size_t offset =
            static_cast<size_t>(t) * embed_dim;

        // Mean
        float mean = 0.0f;

        for (uint32_t d = 0; d < embed_dim; ++d)
            mean += input[offset + d];

        mean /= static_cast<float>(embed_dim);

        // Variance
        float variance = 0.0f;

        for (uint32_t d = 0; d < embed_dim; ++d)
        {
            float diff = input[offset + d] - mean;
            variance += diff * diff;
        }

        variance /= static_cast<float>(embed_dim);

        // Normalize + affine transformation
        float inv_std =
            1.0f / std::sqrt(variance + epsilon);

        for (uint32_t d = 0; d < embed_dim; ++d)
        {
            float normalized =
                (input[offset + d] - mean) * inv_std;

            output[offset + d] =
                normalized * weight.data[d] + bias.data[d];
        }
    }

    return output;
}

std::vector<float> linear(
    const std::vector<float>& input,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t sequence_length,
    uint32_t input_dim,
    uint32_t output_dim)
{
    std::vector<float> output(
        static_cast<size_t>(sequence_length) * output_dim
    );

    for (uint32_t t = 0; t < sequence_length; ++t)
    {
        for (uint32_t o = 0; o < output_dim; ++o)
        {
            float sum = bias.data[o];

            #ifdef __ARM_NEON
            float32x4_t acc = vdupq_n_f32(0.0f);

            uint32_t i = 0;

            for (; i + 3 < input_dim; i += 4)
            {
                float32x4_t x = vld1q_f32(&input[t * input_dim + i]);
                float32x4_t w = vld1q_f32(&weight.data[o * input_dim + i]);

                acc = vmlaq_f32(acc, x, w);
            }

            float tmp[4];
            vst1q_f32(tmp, acc);

            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];

            for (; i < input_dim; ++i)
            {
                sum += input[t * input_dim + i] *
                    weight.data[o * input_dim + i];
            }
            #else
            for (uint32_t i = 0; i < input_dim; ++i)
            {
                sum +=
                    input[t * input_dim + i] *
                    weight.data[o * input_dim + i];
            }
            #endif

            output[t * output_dim + o] = sum;
        }
    }

    return output;
}

std::vector<float> attention(
    const Model& model,
    const std::vector<float>& normalized,
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V)
{
    const uint32_t T = normalized.size() / model.embed_dim;
    const uint32_t C = model.embed_dim;
    const uint32_t H = model.num_heads;
    const uint32_t D = C / H;

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    /*
     * Each head produces:
     *
     *     attention_output [T × D]
     *
     * We store all heads here:
     *
     *     [H × T × D]
     */
    std::vector<float> head_outputs(
        static_cast<size_t>(H) * T * D,
        0.0f
    );

    /*
     * Process each attention head.
     */
    for (uint32_t h = 0; h < H; ++h)
    {
        /*
         * scores[t][j]
         *
         * Query from token t
         * Key   from token j
         */
        std::vector<float> scores(
            static_cast<size_t>(T) * T,
            0.0f
        );

        for (uint32_t t = 0; t < T; ++t)
        {
            for (uint32_t j = 0; j < T; ++j)
            {
                /*
                 * Causal attention:
                 *
                 * token t cannot look at future token j.
                 */
                if (j > t)
                {
                    scores[t * T + j] =
                        -std::numeric_limits<float>::infinity();

                    continue;
                }

                float dot = 0.0f;

                for (uint32_t d = 0; d < D; ++d)
                {
                    /*
                     * Q and K are stored as:
                     *
                     * [T × C]
                     *
                     * Head h occupies:
                     *
                     * h*D ... h*D + D-1
                     */
                    float q =
                        Q[t * C + h * D + d];

                    float k =
                        K[j * C + h * D + d];

                    dot += q * k;
                }

                scores[t * T + j] =
                    dot * scale;
            }

            /*
             * Softmax over all keys j for this query t.
             *
             * softmax(x_i) =
             *     exp(x_i - max(x))
             *     ------------------
             *     sum exp(x_j - max(x))
             *
             * Subtracting max prevents overflow.
             */
            float max_score =
                -std::numeric_limits<float>::infinity();

            for (uint32_t j = 0; j <= t; ++j)
            {
                max_score =
                    std::max(max_score, scores[t * T + j]);
            }

            float sum_exp = 0.0f;

            for (uint32_t j = 0; j <= t; ++j)
            {
                float value =
                    std::exp(scores[t * T + j] - max_score);

                scores[t * T + j] = value;
                sum_exp += value;
            }

            for (uint32_t j = 0; j <= t; ++j)
            {
                scores[t * T + j] /= sum_exp;
            }
        }

        /*
         * Weighted sum:
         *
         * attention = softmax(QKᵀ) V
         *
         * [T × T] × [T × D]
         *       ↓
         *     [T × D]
         */
        for (uint32_t t = 0; t < T; ++t)
        {
            for (uint32_t d = 0; d < D; ++d)
            {
                float value = 0.0f;

                for (uint32_t j = 0; j <= t; ++j)
                {
                    float weight =
                        scores[t * T + j];

                    float v =
                        V[j * C + h * D + d];

                    value += weight * v;
                }

                head_outputs[
                    h * T * D +
                    t * D +
                    d
                ] = value;
            }
        }
    }

    /*
     * Concatenate heads.
     *
     * PyTorch does:
     *
     * attention_output = attention_output.transpose(1, 2)
     * attention_output = attention_output.contiguous()
     *                         .view(B,T,C)
     *
     * We directly construct [T × C].
     */
    std::vector<float> concatenated(
        static_cast<size_t>(T) * C
    );

    for (uint32_t t = 0; t < T; ++t)
    {
        for (uint32_t h = 0; h < H; ++h)
        {
            for (uint32_t d = 0; d < D; ++d)
            {
                concatenated[
                    t * C +
                    h * D +
                    d
                ] =
                    head_outputs[
                        h * T * D +
                        t * D +
                        d
                    ];
            }
        }
    }

    /*
     * Final attention projection:
     *
     * attention_output =
     *     self.attention_proj(attention_output)
     */
    const Tensor& projection_weight =
        find_tensor(
            model,
            "attention_proj.weight"
        );

    const Tensor& projection_bias =
        find_tensor(
            model,
            "attention_proj.bias"
        );

    return linear(
        concatenated,
        projection_weight,
        projection_bias,
        T,
        C,
        C
    );
}

float gelu(float x)
{
    constexpr float inv_sqrt_2 =
        0.7071067811865475f;

    return 0.5f * x *
        (1.0f + std::erf(
            x * inv_sqrt_2
        ));
}

std::vector<float> transformer_block_tail(
    const Model& model,
    const std::vector<float>& input,
    const std::vector<float>& attention_output)
{
    const uint32_t T =
        static_cast<uint32_t>(
            input.size() / model.embed_dim
        );

    const uint32_t C = model.embed_dim;
    const uint32_t F = model.ffn_dim;

    /*
     * First residual:
     *
     * X = X + attention_output
     */
    std::vector<float> X(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        X[i] = input[i] + attention_output[i];

    /*
     * Second LayerNorm
     */
    const Tensor& ln2_weight =
        find_tensor(model, "ln2.weight");

    const Tensor& ln2_bias =
        find_tensor(model, "ln2.bias");

    std::vector<float> X_norm =
        layer_norm(
            X,
            ln2_weight,
            ln2_bias,
            T,
            C
        );

    /*
     * FFN first linear:
     *
     * 64 → 256
     */
    const Tensor& ffn1_weight =
        find_tensor(model, "ffn.0.weight");

    const Tensor& ffn1_bias =
        find_tensor(model, "ffn.0.bias");

    std::vector<float> hidden =
        linear(
            X_norm,
            ffn1_weight,
            ffn1_bias,
            T,
            C,
            F
        );

    /*
     * GELU
     */
    for (float& value : hidden)
        value = gelu(value);

    /*
     * FFN second linear:
     *
     * 256 → 64
     */
    const Tensor& ffn2_weight =
        find_tensor(model, "ffn.2.weight");

    const Tensor& ffn2_bias =
        find_tensor(model, "ffn.2.bias");

    std::vector<float> ffn_output =
        linear(
            hidden,
            ffn2_weight,
            ffn2_bias,
            T,
            F,
            C
        );

    /*
     * Second residual:
     *
     * X = X + FFN(X_norm)
     */
    for (size_t i = 0; i < X.size(); ++i)
        X[i] += ffn_output[i];

    return X;
}

std::vector<float> output_logits(
    const Model& model,
    const std::vector<float>& input)
{
    const uint32_t T =
        static_cast<uint32_t>(
            input.size() / model.embed_dim
        );

    const uint32_t C = model.embed_dim;

    /*
     * Final LayerNorm
     */
    const Tensor& final_norm_weight =
        find_tensor(model, "final_norm.weight");

    const Tensor& final_norm_bias =
        find_tensor(model, "final_norm.bias");

    std::vector<float> normalized =
        layer_norm(
            input,
            final_norm_weight,
            final_norm_bias,
            T,
            C
        );

    /*
     * Final projection:
     *
     * 64 → 65
     *
     * One row of 65 logits for every token.
     */
    const Tensor& output_weight =
        find_tensor(model, "out_proj.weight");

    const Tensor& output_bias =
        find_tensor(model, "out_proj.bias");

    return linear(
        normalized,
        output_weight,
        output_bias,
        T,
        C,
        model.vocab_size
    );
}

std::vector<float> forward(
    const Model& model,
    const std::vector<uint32_t>& token_ids)
{
    std::vector<float> X =
        build_embeddings(model, token_ids);

    const uint32_t T =
        static_cast<uint32_t>(token_ids.size());

    const Tensor& ln1_weight =
        find_tensor(model, "ln1.weight");

    const Tensor& ln1_bias =
        find_tensor(model, "ln1.bias");

    std::vector<float> normalized =
        layer_norm(
            X,
            ln1_weight,
            ln1_bias,
            T,
            model.embed_dim
        );

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
            T,
            model.embed_dim,
            model.embed_dim
        );

    std::vector<float> K =
        linear(
            normalized,
            k_weight,
            k_bias,
            T,
            model.embed_dim,
            model.embed_dim
        );

    std::vector<float> V =
        linear(
            normalized,
            v_weight,
            v_bias,
            T,
            model.embed_dim,
            model.embed_dim
        );

    std::vector<float> attention_output =
        attention(
            model,
            normalized,
            Q,
            K,
            V
        );

    X =
        transformer_block_tail(
            model,
            X,
            attention_output
        );

    return output_logits(model, X);
}
