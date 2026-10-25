import torch

from transformer_block_dev import (
    model,
    vocab_size,
    context_length,
    embed_dim,
    ffn_dim,
)

state = model.state_dict()

print("Exporting model...")
print(f"vocab_size     = {vocab_size}")
print(f"context_length = {context_length}")
print(f"embed_dim      = {embed_dim}")
print(f"ffn_dim        = {ffn_dim}")

for name, tensor in state.items():
    print(f"{name:35s} {tuple(tensor.shape)}")

# Convert everything to FP32 CPU tensors.
exported = {
    name: tensor.detach().cpu().float()
    for name, tensor in state.items()
}

torch.save(
    {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "embed_dim": embed_dim,
        "ffn_dim": ffn_dim,
        "num_heads": model.num_heads,
        "state_dict": exported,
    },
    "yatt_model.pt",
)

print("\nSaved deployment model to yatt_model.pt")