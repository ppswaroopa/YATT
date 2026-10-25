from pathlib import Path

import torch


class TextDataset:

    def __init__(
        self,
        path,
        tokenizer,
    ):
        text = Path(path).read_text()

        self.text = text
        self.tokenizer = tokenizer

        self.data = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.data)

    def get_batch(
        self,
        batch_size,
        context_length,
    ):
        starts = torch.randint(
            0,
            len(self.data) - context_length,
            (batch_size,),
        )

        x = torch.stack([
            self.data[
                i:i + context_length
            ]
            for i in starts
        ])

        y = torch.stack([
            self.data[
                i + 1:i + context_length + 1
            ]
            for i in starts
        ])

        return x, y
