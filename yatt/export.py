from pathlib import Path
import struct
import torch


CHECKPOINT_PATH = "checkpoints/tiny_transformer.pt"
EXPORT_PATH = "checkpoints/yatt_model.bin"

MAGIC = b"YATT"
VERSION = 1
DTYPE_FLOAT32 = 1


def write_u32(f, value):
    f.write(struct.pack("<I", value))


def write_u64(f, value):
    f.write(struct.pack("<Q", value))


def write_tensor_metadata(f, name, tensor):
    name_bytes = name.encode("utf-8")

    write_u32(f, len(name_bytes))
    f.write(name_bytes)

    write_u32(f, tensor.ndim)

    for dimension in tensor.shape:
        write_u32(f, dimension)

    write_u32(f, DTYPE_FLOAT32)

    data_bytes = tensor.numel() * 4
    write_u64(f, data_bytes)


def export_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]

    tensors = []

    for name, tensor in state_dict.items():
        if name == "causal_mask":
            continue

        tensor = tensor.detach().cpu().float().contiguous()
        tensors.append((name, tensor))

    with open(EXPORT_PATH, "wb") as f:

        # -------------------------
        # Header
        # -------------------------

        f.write(MAGIC)

        write_u32(f, VERSION)
        write_u32(f, checkpoint["vocab_size"])
        write_u32(f, checkpoint["context_length"])
        write_u32(f, checkpoint["embed_dim"])
        write_u32(f, checkpoint["ffn_dim"])
        write_u32(f, checkpoint["num_heads"])
        write_u32(f, len(tensors))

        # -------------------------
        # Tensor metadata
        # -------------------------

        for name, tensor in tensors:
            write_tensor_metadata(f, name, tensor)

        # -------------------------
        # Tensor data
        # -------------------------

        for name, tensor in tensors:
            data = tensor.numpy().tobytes()
            f.write(data)

    print(f"Exported model to {EXPORT_PATH}")

    total_bytes = Path(EXPORT_PATH).stat().st_size
    print(f"Model size: {total_bytes} bytes ({total_bytes / 1024:.2f} KiB)")


if __name__ == "__main__":
    export_model()