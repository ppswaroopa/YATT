import torch
import torch.nn as nn


class TinyTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embed_dim,
        ffn_dim,
        num_heads=4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.head_dim = embed_dim // num_heads

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            context_length,
            embed_dim,
        )

        # Pre-attention normalization
        self.ln1 = nn.LayerNorm(embed_dim)

        # Attention projections
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

        # Pre-FFN normalization
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
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

        # Final normalization
        self.final_norm = nn.LayerNorm(embed_dim)

        # Convert representation → vocabulary logits
        self.out_proj = nn.Linear(
            embed_dim,
            vocab_size,
        )

        # Causal mask
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

    def forward(self, x):

        B, T = x.shape

        # Token + position embeddings
        token_vectors = self.token_embedding(x)

        positions = torch.arange(
            T,
            device=x.device,
        )

        position_vectors = self.position_embedding(
            positions
        )

        X = token_vectors + position_vectors

        # --------------------------------------------------------
        # Self-attention
        # --------------------------------------------------------

        X_norm = self.ln1(X)

        Q = self.q_proj(X_norm)
        K = self.k_proj(X_norm)
        V = self.v_proj(X_norm)

        # [B, T, C] → [B, T, heads, head_dim]
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

        # [B, T, heads, head_dim]
        # → [B, heads, T, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        scores = scores.masked_fill(
            ~self.causal_mask[:T, :T],
            float("-inf"),
        )

        # Attention probabilities
        weights = torch.softmax(
            scores,
            dim=-1,
        )

        # Weighted values
        attention_output = weights @ V

        # [B, heads, T, head_dim]
        # → [B, T, heads, head_dim]
        attention_output = attention_output.transpose(
            1,
            2,
        )

        # [B, T, heads, head_dim]
        # → [B, T, embed_dim]
        attention_output = attention_output.contiguous().view(
            B,
            T,
            self.embed_dim,
        )

        # Mix information from all heads
        attention_output = self.attention_proj(
            attention_output
        )

        # First residual
        X = X + attention_output

        # --------------------------------------------------------
        # Feed-forward network
        # --------------------------------------------------------

        X_norm = self.ln2(X)

        ffn_output = self.ffn(X_norm)

        # Second residual
        X = X + ffn_output

        # Final normalization
        X = self.final_norm(X)

        # Vocabulary logits
        logits = self.out_proj(X)

        return logits
