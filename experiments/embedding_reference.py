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

    token_vectors = model.token_embedding(tokens)

    positions = torch.arange(len(token_ids))
    position_vectors = model.position_embedding(positions)

    X = token_vectors[0] + position_vectors

for t, token_id in enumerate(token_ids):
    print(
        f"token {token_id}: "
        + ", ".join(f"{v:.6f}" for v in X[t, :5])
    )
