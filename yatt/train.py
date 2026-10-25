from pathlib import Path

import torch
import torch.nn.functional as F

from yatt.model import TinyTransformer
from yatt.tokenizer import CharTokenizer
from yatt.dataset import TextDataset


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_PATH = "data/input.txt"
CHECKPOINT_PATH = "checkpoints/tiny_transformer.pt"

BATCH_SIZE = 32
CONTEXT_LENGTH = 64

EMBED_DIM = 64
FFN_DIM = 256
NUM_HEADS = 4

LEARNING_RATE = 3e-4
NUM_STEPS = 2000


def train():

    torch.manual_seed(42)

    # Dataset
    text = Path(DATA_PATH).read_text()

    tokenizer = CharTokenizer(text)

    dataset = TextDataset(
        DATA_PATH,
        tokenizer,
    )

    vocab_size = tokenizer.vocab_size

    # Model
    model = TinyTransformer(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
    )

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        num_parameters,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Training
    model.train()

    for step in range(NUM_STEPS):

        x, y = dataset.get_batch(
            BATCH_SIZE,
            CONTEXT_LENGTH,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 100 == 0:
            print(
                f"step={step:4d} "
                f"loss={loss.item():.4f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "context_length": CONTEXT_LENGTH,
            "embed_dim": EMBED_DIM,
            "ffn_dim": FFN_DIM,
            "num_heads": NUM_HEADS,
            "chars": tokenizer.chars,
        },
        CHECKPOINT_PATH,
    )

    print(
        f"\nSaved model to {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    train()
