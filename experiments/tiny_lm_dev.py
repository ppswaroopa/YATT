import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# --------------------------------------------------
# Tiny language-model setup
# --------------------------------------------------

vocab_size = 65
embed_dim = 4
seq_len = 4

# A tiny sequence of token IDs.
# Think of this as four characters from Shakespeare.
x = torch.tensor([18, 47, 56, 57])

# The correct next token for each position.
#
# x: [18, 47, 56, 57]
# y: [47, 56, 57, 58]
#
# In other words:
# 18 -> 47
# 47 -> 56
# 56 -> 57
# 57 -> 58
y = torch.tensor([47, 56, 57, 58])

print("x:", x)
print("y:", y)

# --------------------------------------------------
# Embedding
# --------------------------------------------------

embedding = nn.Embedding(vocab_size, embed_dim)

X = embedding(x)

print("\nX:")
print(X)

print("X shape:", X.shape)

# --------------------------------------------------
# Q / K / V
# --------------------------------------------------

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
# Self-attention
# --------------------------------------------------

scores = Q @ K.T
scores = scores / (embed_dim ** 0.5)

# Causal mask
mask = torch.tril(torch.ones(seq_len, seq_len))

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

weights = torch.softmax(scores, dim=-1)

print("\nAttention weights:")
print(weights)

# Contextualized representations
attention_output = weights @ V

print("\nAttention output:")
print(attention_output)

print("Attention output shape:", attention_output.shape)

# --------------------------------------------------
# Predict next token
# --------------------------------------------------

# Convert each 4-dimensional representation
# into 65 scores — one for every possible character.
W_out = nn.Parameter(torch.randn(embed_dim, vocab_size))

logits = attention_output @ W_out

print("\nLogits:")
print(logits)

print("Logits shape:", logits.shape)

# --------------------------------------------------
# Language-model loss
# --------------------------------------------------

loss = F.cross_entropy(logits, y)

print("\nLoss:")
print(loss.item())

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

parameters = [
    embedding.weight,
    Wq,
    Wk,
    Wv,
    W_out,
]

optimizer = torch.optim.Adam(parameters, lr=0.01)

# --------------------------------------------------
# Training loop
# --------------------------------------------------

for step in range(1000):

    # Forward pass
    X = embedding(x)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = Q @ K.T
    scores = scores / (embed_dim ** 0.5)

    scores = scores.masked_fill(
        mask == 0,
        float("-inf")
    )

    weights = torch.softmax(scores, dim=-1)

    attention_output = weights @ V

    logits = attention_output @ W_out

    # Calculate loss
    loss = F.cross_entropy(logits, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if step % 100 == 0:
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == y).sum().item()

        print(
            f"step={step:4d} "
            f"loss={loss.item():.4f} "
            f"correct={correct}/4"
        )

# --------------------------------------------------
# Inspect learned attention
# --------------------------------------------------

with torch.no_grad():

    X = embedding(x)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    scores = Q @ K.T
    scores = scores / (embed_dim ** 0.5)

    scores = scores.masked_fill(
        mask == 0,
        float("-inf")
    )

    learned_weights = torch.softmax(scores, dim=-1)

print("\nLearned attention weights:")
print(learned_weights)

print("\nRow sums:")
print(learned_weights.sum(dim=-1))
