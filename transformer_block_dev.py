from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)


# ============================================================
# Dataset
# ============================================================

text = Path("data/input.txt").read_text()

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(
    encode(text),
    dtype=torch.long,
)


# ============================================================
# Hyperparameters
# ============================================================

batch_size = 32
context_length = 64

vocab_size = len(chars)
embed_dim = 64
ffn_dim = 256

learning_rate = 3e-4
num_steps = 2000


# ============================================================
# Batch sampling
# ============================================================

def get_batch():
    starts = torch.randint(
        0,
        len(data) - context_length,
        (batch_size,),
    )

    x = torch.stack([
        data[i:i + context_length]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + context_length + 1]
        for i in starts
    ])

    return x, y


# ============================================================
# Transformer Model
# ============================================================

class TinyTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim,
        ffn_dim,
    ):
        super().__init__()

        self.num_heads = 4
        assert embed_dim % self.num_heads == 0
        self.head_dim = embed_dim // self.num_heads

        # ----------------------------------------------------
        # Token + position embeddings
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim,
        )

        # ----------------------------------------------------
        # First LayerNorm
        # ----------------------------------------------------

        self.ln1 = nn.LayerNorm(embed_dim)

        # ----------------------------------------------------
        # Self-attention projections
        # ----------------------------------------------------

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.attention_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        # ----------------------------------------------------
        # Second LayerNorm
        # ----------------------------------------------------

        self.ln2 = nn.LayerNorm(embed_dim)

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        self.ffn = nn.Sequential(
            nn.Linear(
                embed_dim,
                ffn_dim,
            ),

            nn.GELU(),

            nn.Linear(
                ffn_dim,
                embed_dim,
            ),
        )

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(embed_dim)

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            embed_dim,
            vocab_size,
        )

        # ----------------------------------------------------
        # Causal attention mask
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
        )

    # ========================================================
    # Forward pass
    # ========================================================

    def forward(self, x):

        B, T = x.shape

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_vectors = self.token_embedding(x)

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=x.device,
        )

        position_vectors = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Token + position
        # ----------------------------------------------------

        X = token_vectors + position_vectors

        # ====================================================
        # Self-attention
        # ====================================================

        X_norm = self.ln1(X)

        Q = self.q_proj(X_norm)
        K = self.k_proj(X_norm)
        V = self.v_proj(X_norm)

        # ------------------------------------------------------------
        # Split embedding dimension across attention heads
        #
        # Before:
        #     [B, T, 64]
        #
        # After:
        #     [B, T, 4, 16]
        # ------------------------------------------------------------

        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )

        # ------------------------------------------------------------
        # Move heads before sequence dimension
        #
        # [B, T, heads, head_dim]
        #         ↓
        # [B, heads, T, head_dim]
        # ------------------------------------------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ------------------------------------------------------------
        # Attention scores
        #
        # Q:
        #     [B, heads, T, head_dim]
        #
        # K:
        #     [B, heads, T, head_dim]
        #
        # K.transpose(-2, -1):
        #     [B, heads, head_dim, T]
        #
        # result:
        #     [B, heads, T, T]
        # ------------------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        # ------------------------------------------------------------
        # Causal mask
        # ------------------------------------------------------------

        scores = scores.masked_fill(
            ~self.causal_mask[:T, :T],
            float("-inf"),
        )

        # ------------------------------------------------------------
        # Attention probabilities
        # ------------------------------------------------------------

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        # ------------------------------------------------------------
        # Weighted values
        #
        # [B, heads, T, T]
        #              @
        # [B, heads, T, head_dim]
        #
        # =
        #
        # [B, heads, T, head_dim]
        # ------------------------------------------------------------

        attention_output = weights @ V

        # ------------------------------------------------------------
        # Put sequence dimension before heads
        #
        # [B, heads, T, head_dim]
        #          ↓
        # [B, T, heads, head_dim]
        # ------------------------------------------------------------

        attention_output = attention_output.transpose(1, 2)

        # ------------------------------------------------------------
        # Combine heads
        #
        # [B, T, heads, head_dim]
        #          ↓
        # [B, T, embed_dim]
        #
        # 4 × 16 = 64
        # ------------------------------------------------------------

        attention_output = attention_output.contiguous().view(
            B,
            T,
            embed_dim,
        )

        # ------------------------------------------------------------
        # Mix information from all heads
        # ------------------------------------------------------------

        attention_output = self.attention_proj(
            attention_output
        )

        # ----------------------------------------------------
        # First residual
        # ----------------------------------------------------

        X = X + attention_output

        # ====================================================
        # Feed-forward network
        # ====================================================

        X_norm = self.ln2(X)

        ffn_output = self.ffn(X_norm)

        # ----------------------------------------------------
        # Second residual
        # ----------------------------------------------------

        X = X + ffn_output

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        X = self.final_norm(X)

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        logits = self.out_proj(X)

        # print("Q:", Q.shape)
        # print("K:", K.shape)
        # print("V:", V.shape)
        # print("Attention weights:", weights.shape)
        # print("Attention output:", attention_output.shape)

        return logits


# ============================================================
# Create model
# ============================================================

model = TinyTransformer(
    vocab_size=vocab_size,
    context_length=context_length,
    embed_dim=embed_dim,
    ffn_dim=ffn_dim,
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


# ============================================================
# Parameter count
# ============================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", num_parameters)


# ============================================================
# Training
# ============================================================

model.train()

for step in range(num_steps):

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    x, y = get_batch()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits = model(x)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if step % 100 == 0:
        print(
            f"step={step:4d} "
            f"loss={loss.item():.4f}"
        )


# ============================================================
# Autoregressive Generation
# ============================================================

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens,
    temperature=1.0,
):

    model.eval()

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    tokens = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
    )

    # tokens shape:
    #
    # [1, prompt_length]
    #

    for _ in range(max_new_tokens):

        # ----------------------------------------------------
        # Keep only the most recent context_length tokens
        #
        # This becomes important once generated text is
        # longer than the model's context window.
        # ----------------------------------------------------

        context = tokens[:, -context_length:]

        # ----------------------------------------------------
        # Forward pass
        #
        # [1, T]
        #     ↓
        # [1, T, vocab_size]
        # ----------------------------------------------------

        logits = model(context)

        # ----------------------------------------------------
        # We only care about the prediction for the LAST
        # token position.
        #
        # [1, T, vocab_size]
        #          ↓
        # [1, vocab_size]
        # ----------------------------------------------------

        next_token_logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Greedy decoding:
        #
        # choose the token with the highest probability.
        #
        # argmax(logits) is equivalent to argmax(softmax(logits))
        # ----------------------------------------------------

        probabilities = torch.softmax(
	    next_token_logits / temperature,
	    dim=-1,
	)

        next_token = torch.multinomial(
	    probabilities,
	    num_samples=1,
	)
        # ----------------------------------------------------
        # Append generated token
        #
        # [1, T] + [1, 1]
        #     ↓
        # [1, T+1]
        # ----------------------------------------------------

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    # --------------------------------------------------------
    # Decode the complete sequence
    # --------------------------------------------------------

    return decode(
        tokens[0].tolist()
    )


# ============================================================
# Generate text
# ============================================================

prompt = "First Citizen:"

generated_text = generate(
    model,
    prompt,
    max_new_tokens=300,
    temperature=1.0,
)

print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print(generated_text)

torch.save(model.state_dict(), "tiny_transformer.pt")
print("\nSaved model to tiny_transformer.pt")
