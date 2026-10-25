import torch

from yatt.model import TinyTransformer

checkpoint = torch.load(
    "checkpoints/tiny_transformer.pt",
    map_location="cpu"
)

model = TinyTransformer(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    embed_dim=checkpoint["embed_dim"],
    ffn_dim=checkpoint["ffn_dim"],
    num_heads=checkpoint["num_heads"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

chars = checkpoint["chars"]
stoi = {ch: i for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text]

prefix = "First Citizen:"

tokens = encode(prefix)

with torch.no_grad():
    for step in range(10):
        x = torch.tensor([tokens], dtype=torch.long)

        logits = model(x)

        last_logits = logits[0, -1]

        values, indices = torch.topk(last_logits, 5)

        print(f"step {step} top predictions:", end=" ")

        for token_id, value in zip(indices.tolist(), values.tolist()):
            print(f"({token_id}, {value:.5f})", end=" ")

        print()

        next_token = torch.argmax(last_logits).item()

        print(f"step {step}: token={next_token}")

        tokens.append(next_token)
