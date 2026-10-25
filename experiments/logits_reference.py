import torch
from yatt.model import TinyTransformer

CHECKPOINT = "checkpoints/tiny_transformer.pt"

token_ids = [
    18, 47, 56, 57, 58, 1, 15,
    47, 58, 47, 64, 43, 52, 10
]

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
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

with torch.no_grad():
    tokens = torch.tensor([token_ids])

    logits = model(tokens)

    last_logits = logits[0, -1]

for i, value in enumerate(last_logits):
    print(f"{i}: {value.item():.6f}")
