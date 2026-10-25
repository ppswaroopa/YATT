import torch

from yatt.model import TinyTransformer


def generate(
    model,
    tokenizer,
    prompt,
    context_length,
    max_new_tokens,
    temperature=1.0,
):

    model.eval()

    tokens = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            context = tokens[
                :, -context_length:
            ]

            logits = model(context)

            next_token_logits = logits[
                :, -1, :
            ]

            probabilities = torch.softmax(
                next_token_logits / temperature,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

    return tokenizer.decode(
        tokens[0].tolist()
    )
