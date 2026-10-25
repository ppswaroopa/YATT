from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

text = Path("data/input.txt").read_text()

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# --------------------------------------------------
# Batch
# --------------------------------------------------

batch_size = 32
context_length = 64

starts = torch.randint(
    0,
    len(data) - context_length,
    (batch_size,)
)

x = torch.stack([
    data[i:i + context_length]
    for i in starts
])

y = torch.stack([
    data[i + 1:i + context_length + 1]
    for i in starts
])

print("x shape:", x.shape)
print("y shape:", y.shape)

print("\nFirst input:")
print(decode(x[0].tolist()))

print("\nFirst target:")
print(decode(y[0].tolist()))

print("\nShift check:")
print(torch.equal(y[:, :-1], x[:, 1:]))

vocab_size = len(chars)
embed_dim = 64

embedding = nn.Embedding(vocab_size, embed_dim)

X = embedding(x)

print("\nVocabulary size:", vocab_size)
print("X shape:", X.shape)

q_proj = nn.Linear(embed_dim, embed_dim)
k_proj = nn.Linear(embed_dim, embed_dim)
v_proj = nn.Linear(embed_dim, embed_dim)

Q = q_proj(X)
K = k_proj(X)
V = v_proj(X)

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

scores = Q @ K.transpose(-2, -1)

scores = scores / (embed_dim ** 0.5)

mask = torch.tril(
    torch.ones(context_length, context_length)
)

scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)

weights = torch.softmax(scores, dim=-1)

output = weights @ V

print("\nScores shape:", scores.shape)
print("Weights shape:", weights.shape)
print("Attention output shape:", output.shape)

out_proj = nn.Linear(embed_dim, vocab_size)

logits = out_proj(output)

print("\nLogits shape:", logits.shape)

print("\nActivation statistics:")
print("X:", X.min().item(), X.max().item(), X.std().item())
print("Q:", Q.min().item(), Q.max().item(), Q.std().item())
print("K:", K.min().item(), K.max().item(), K.std().item())
print("V:", V.min().item(), V.max().item(), V.std().item())
print("scores:", scores[scores != float("-inf")].std().item())
print("output:", output.min().item(), output.max().item(), output.std().item())
print("logits:", logits.min().item(), logits.max().item(), logits.std().item())

loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    y.reshape(-1)
)

print("\nLoss:", loss.item())


print("\nParameter shapes:")
print("q_proj:", q_proj.weight.shape)
print("k_proj:", k_proj.weight.shape)
print("v_proj:", v_proj.weight.shape)
print("out_proj:", out_proj.weight.shape)
