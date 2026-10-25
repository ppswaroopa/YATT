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

    X = token_vectors + position_vectors

    X_norm = model.ln1(X)

    Q = model.q_proj(X_norm)
    K = model.k_proj(X_norm)
    V = model.v_proj(X_norm)

    B, T, C = Q.shape
    H = model.num_heads
    D = model.head_dim

    Q = Q.view(B, T, H, D).transpose(1, 2)
    K = K.view(B, T, H, D).transpose(1, 2)
    V = V.view(B, T, H, D).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1)
    scores = scores / (D ** 0.5)

    mask = model.causal_mask[:T, :T]
    scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    attention_output = weights @ V

    attention_output = attention_output.transpose(1, 2)
    attention_output = attention_output.contiguous().view(B, T, C)

    attention_output = model.attention_proj(attention_output)

for t, token_id in enumerate(token_ids):
    print(
        f"token {token_id}: "
        + ", ".join(
            f"{v:.6f}"
            for v in attention_output[0, t, :5]
        )
    )
