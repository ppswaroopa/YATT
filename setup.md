# Training Machine Setup

## Setup python environment

```bash
sudo apt update
sudo apt install -y python3-venv

mkdir yatt
python3 -m venv .yatt
source .yatt/bin/activate
python -m pip install --upgrade pip

```

```bash
# Use https://pytorch.org/get-started/locally and install Pytorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Run this to check if installation was OK
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

-----
PyTorch: 2.14.1+cu132
CUDA available: True
CUDA: 13.2
GPU: NVIDIA GeForce GTX 1660 Ti
-----
```

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

z = x @ y

print(device)
print(z.shape)
print(z.device)
```

```text
cuda
torch.Size([4096, 4096])
cuda:0
```

### Download TinyShakesphere

```bash
mkdir data
cd data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

