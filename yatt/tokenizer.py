class CharTokenizer:

    def __init__(self, text):
        chars = sorted(set(text))

        self.chars = chars
        self.stoi = {
            ch: i
            for i, ch in enumerate(chars)
        }
        self.itos = {
            i: ch
            for i, ch in enumerate(chars)
        }

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, ids):
        return "".join(
            self.itos[i]
            for i in ids
        )
