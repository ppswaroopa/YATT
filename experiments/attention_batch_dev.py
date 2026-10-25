import torch
import torch.nn as nn

torch.manual_seed(42)

# --------------------------------------------------
# Dimensions
# --------------------------------------------------

batch_size = 32
seq_len = 64
vocab_size = 65
embed_dim = 64

# --------------------------------------------------
# Fake token batch
# --------------------------------------------------

x = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len)
)

print("x shape:", x.shape)

# --------------------------------------------------
# Embedding
# --------------------------------------------------

embedding = nn.Embedding(vocab_size, embed_dim)

X = embedding(x)

print("X shape:", X.shape)

# --------------------------------------------------
# Q / K / V
# --------------------------------------------------

Wq = torch.randn(embed_dim, embed_dim)
Wk = torch.randn(embed_dim, embed_dim)
Wv = torch.randn(embed_dim, embed_dim)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# --------------------------------------------------
# Batched attention scores
# --------------------------------------------------

scores = Q @ K.transpose(-2, -1)

print("\nScores shape:", scores.shape)

# --------------------------------------------------
# Causal mask
# --------------------------------------------------

mask = torch.tril(
    torch.ones(seq_len, seq_len)
)

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

print("Masked scores shape:", scores.shape)

# --------------------------------------------------
# Softmax
# --------------------------------------------------

weights = torch.softmax(scores, dim=-1)

print("Attention weights shape:", weights.shape)

# --------------------------------------------------
# Weighted values
# --------------------------------------------------

output = weights @ V

print("Attention output shape:", output.shape)
