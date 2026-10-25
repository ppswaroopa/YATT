import torch
import torch.nn as nn

torch.manual_seed(42)

# --------------------------------------------------
# Tiny example
# --------------------------------------------------

seq_len = 4
embed_dim = 4

# Imagine these came from the embedding layer.
# 4 tokens, each represented by 4 numbers.
X = torch.randn(seq_len, embed_dim)

print("X:")
print(X)
print("X shape:", X.shape)


# --------------------------------------------------
# Learned Q, K, V projections
# --------------------------------------------------

# Wq = torch.randn(embed_dim, embed_dim)
# Wk = torch.randn(embed_dim, embed_dim)
# Wv = torch.randn(embed_dim, embed_dim)

Wq = nn.Parameter(torch.randn(embed_dim, embed_dim))
Wk = nn.Parameter(torch.randn(embed_dim, embed_dim))
Wv = nn.Parameter(torch.randn(embed_dim, embed_dim))

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# --------------------------------------------------
# Attention scores
# --------------------------------------------------

scores = Q @ K.T

print("\nRaw attention scores:")
print(scores)
print("Scores shape:", scores.shape)


# --------------------------------------------------
# Scale scores
# --------------------------------------------------

scores = scores / (embed_dim ** 0.5)


# --------------------------------------------------
# Causal mask
# --------------------------------------------------

mask = torch.tril(torch.ones(seq_len, seq_len))

scores = scores.masked_fill(mask == 0, float("-inf"))

print("\nMasked scores:")
print(scores)


# --------------------------------------------------
# Convert scores to attention weights
# --------------------------------------------------

weights = torch.softmax(scores, dim=-1)

print("\nAttention weights:")
print(weights)

print("\nRow sums:")
print(weights.sum(dim=-1))


# --------------------------------------------------
# Weighted combination of Values
# --------------------------------------------------

output = weights @ V

print("\nAttention output:")
print(output)

print("Output shape:", output.shape)


# --------------------------------------------------
# Tiny learning objective
# --------------------------------------------------

# Pretend we want the attention output to move
# toward this target representation.
target = torch.randn_like(output)

loss = ((output - target) ** 2).mean()

print("\nLoss:")
print(loss.item())

# Backpropagation
loss.backward()

print("\nGradients:")

print("Wq.grad:")
print(Wq.grad)

print("\nWk.grad:")
print(Wk.grad)

print("\nWv.grad:")
print(Wv.grad)


print("\n--- Manual token 3 calculation ---")

i = 2

print("Attention weights for token 3:")
print(weights[i])

print("\nV:")
print(V)

manual_output = (
    weights[i, 0] * V[0]
    + weights[i, 1] * V[1]
    + weights[i, 2] * V[2]
    + weights[i, 3] * V[3]
)

print("\nManual output:")
print(manual_output)

print("\nAutomatic output:")
print(output[i])

print("\nSame:", torch.allclose(manual_output, output[i]))
