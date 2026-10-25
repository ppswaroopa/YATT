import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

z = x @ y

print(device)
print(z.shape)
print(z.device)
