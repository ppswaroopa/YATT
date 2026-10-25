from pathlib import Path
import torch

text = Path("data/input.txt").read_text()

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 64

x = data[:block_size]
y = data[1:block_size + 1]

print("Text length:", len(text))
print("Vocabulary size:", len(chars))
print("Data shape:", data.shape)
print("Data dtype:", data.dtype)

print("\nFirst 100 token IDs:")
print(data[:100])

print("\nDecoded:")
print(decode(data[:100].tolist()))

print("x shape:", x.shape)
print("y shape:", y.shape)

print("\nx:")
print(x)

print("\ny:")
print(y)

print("\nDecoded x:")
print(decode(x.tolist()))

print("\nDecoded y:")
print(decode(y.tolist()))
