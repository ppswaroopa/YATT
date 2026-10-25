from pathlib import Path
import torch

# --------------------------------------------------
# Load and tokenize Tiny Shakespeare
# --------------------------------------------------

text = Path("data/input.txt").read_text()

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

print("Data shape:", data.shape)
print("Vocabulary size:", len(chars))

# --------------------------------------------------
# Random batch sampling
# --------------------------------------------------

batch_size = 32
context_length = 64

torch.manual_seed(42)

starts = torch.randint(
    0,
    len(data) - context_length,
    (batch_size,)
)

x = torch.stack([
    data[i : i + context_length]
    for i in starts
])

y = torch.stack([
    data[i + 1 : i + context_length + 1]
    for i in starts
])

print("\nBatch:")
print("x shape:", x.shape)
print("y shape:", y.shape)

print("\nFirst example:")
print("x:", repr(decode(x[0].tolist())))
print("y:", repr(decode(y[0].tolist())))

print("\nShift check:")
print(torch.equal(y[0, :-1], x[0, 1:]))
