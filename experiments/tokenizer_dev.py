from pathlib import Path

text = Path("data/input.txt").read_text()

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

sample = "First Citizen:"

encoded = encode(sample)

print("\nOriginal:")
print(sample)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(decode(encoded))

print(f"Characters in corpus: {len(text):,}")
print(f"Unique characters: {len(chars)}")
print("Characters:")
print(chars)
