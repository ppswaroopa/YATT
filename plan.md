# Tiny Transformer

## Goal

Build, train, deploy, and benchmark a **tiny-Transformer language
model** and run it on the BeagleBone Black.

The goal is to learn the complete embedded-AI pipeline:

``` text
dataset
  -> tokenizer
  -> tiny Transformer training
  -> export
  -> compact inference format
  -> ARMv7 deployment
  -> inference runtime
  -> profiling
  -> optimization
```

The stretch goal is to make the final inference runtime mostly/all
native C++ and use the BeagleBone's ARM NEON SIMD instructions for at
least one hot kernel.

## 1. Target hardware

### BeagleBone Black - Rev C

Target:

- BeagleBone Black
- TI Sitara AM3358
- ARM Cortex-A8
- 1 GHz
- ARMv7 32-bit
- NEON SIMD / floating-point accelerator
- 512 MB DDR3/DDR3L RAM
- 4 GB eMMC
- 10/100 Ethernet
- Debian Linux
- Architecture: `armv7l`
- OS: Debian GNU/Linux 10 (buster)
- Kernel: `4.19.94-ti-r42`

### Important constraints

The BBB is intentionally hostile hardware for modern AI:

- 512 MB RAM
- 32-bit userspace/architecture
- 1 GHz single Cortex-A8
- no NPU
- no CUDA
- no modern GPU compute
- no large model runtime expected to be practical

## 2. Project Goal

## Minimum success

- Train a tiny Transformer on a host device.
- Save a checkpoint.
- Export the trained model into a simple deployment format.
- Cross-compile an inference program for ARMv7.
- Copy it to the BBB.
- Run inference on the BBB.
- Generate multiple tokens autoregressively.
- Measure inference latency and memory use.

## Strong success

- INT8 or another compact weight representation.
- Native C++ inference.
- No PyTorch/TensorFlow dependency on the BBB.
- Quantitative benchmark of:
  - model size
  - RSS
  - prompt processing time
  - per-token latency
  - tokens/sec
  - CPU utilization

## Stretch success

- Implement at least one optimized matrix/vector kernel using ARM
  NEON.
- Compare:
  - reference scalar implementation
  - NEON implementation
- Produce a measurable speedup.
- Run the inference service as a systemd service.
- Expose a tiny local CLI or HTTP/MQTT interface.

## 3. Recommended model

Start with a deliberately tiny decoder-only Transformer.

### Baseline configuration

``` text
context_length   = 64
embedding_dim    = 64
num_layers       = 2
num_heads        = 4
head_dim         = 16
FFN dimension    = 128 or 256
activation       = GELU or ReLU
parameters       = roughly sub-million to low-million range
```

### Why this size?

It is large enough to contain the important Transformer machinery:

- token embeddings
- positional information
- Q/K/V projections
- multi-head self-attention
- causal masking
- feed-forward network
- residual connections
- normalization
- output projection
- autoregressive sampling

But it is small enough that the BBB can actually execute it.

## 4. Dataset choice

Pick a constrained corpus that makes the output visibly learnable.

### Option A --- Tiny Shakespeare

Recommended first choice.

Advantages:

- small
- easy to obtain
- character/token language modeling works well
- recognizable generated output
- classic Transformer tutorial dataset

Potential target:

``` text
Input:
"ROMEO:"

Output:
"ROMEO: But soft, what light..."
```

### Option B --- Project-specific corpus

Create a small embedded/robotics corpus:

``` text
ROS
MQTT
Linux commands
systemd
robot commands
sensor messages
short technical text
```

This is more personally interesting but less useful for validating
language-model behavior.

## 5. Tokenizer choice

## Preferred first implementation: character-level

Vocabulary is just characters.

Example:

``` text
"hello world"
        ->
['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
```

Advantages:

- tiny vocabulary
- tokenizer is trivial
- easy to reproduce in C++
- no dependency on a tokenizer library
- excellent for understanding the entire pipeline

Target:

``` text
vocab <= 256
```

This makes the deployment format extremely simple.

## Later experiment

Move to a small BPE/token tokenizer.

This is an optional second stage.

## 6. Training stack

Train on the laptop.

Suggested:

- Python
- PyTorch
- CPU or CUDA depending on available machine
- ordinary training loop

Training code should be intentionally readable.

Core components:

``` text
Dataset
  ↓
Embedding
  ↓
Positional embedding
  ↓
TransformerBlock
  ├── LayerNorm
  ├── CausalSelfAttention
  ├── residual
  ├── LayerNorm
  ├── MLP
  └── residual
  ↓
LM head
  ↓
Cross entropy
```

## 7. Architecture decision: pre-LN vs post-LN

Use **pre-LayerNorm** for the first implementation.

Conceptually:

``` text
x
 |
 +----> LayerNorm -> Attention -> + 
 |                               |
 +-------------------------------+
 |
 +----> LayerNorm -> MLP       -> +
                                  |
 +--------------------------------+
```

This is simple and generally easier to train reliably.

## 8. Training target

The model performs next-token prediction.

For:

``` text
HELLO WORLD
```

training pairs are approximately:

``` text
input:  HELLO WORL
target: ELLO WORLD
```

Loss:

``` text
CrossEntropy(logits, target)
```

Use teacher forcing during training.

## 9. First milestone: prove the model works on the laptop

### Test 1

Overfit a tiny dataset.

Expected:

``` text
training loss -> very low
```

Generated output should reproduce training-like sequences.

### Test 2

Train on the complete small corpus.

Save:

``` text
checkpoint.pt
```

### Test 3

Generate text from several prompts.

Save examples to:

``` text
artifacts/generation.txt
```

This becomes the reference against which the embedded implementation is
tested.

## 10. Deployment format

Do NOT deploy a PyTorch checkpoint to the BBB.

Create a deliberately simple model format.

Possible layout:

``` text
model.bin

header:
    magic
    version
    vocab_size
    context_length
    embedding_dim
    num_layers
    num_heads
    ffn_dim
    dtype

weights:
    token_embedding
    position_embedding
    layer_0...
    layer_1...
    lm_head
```

Initially use FP32.

Then add INT8 later.

### Important design principle

The model file should contain **only what inference needs**.

No optimizer state.

No gradients.

No training metadata beyond what the runtime requires.

## 11. Inference runtime

Implement a minimal native C++ runtime.

Suggested layout:

``` text
bbb-transformer/
├── CMakeLists.txt
├── include/
│   ├── tensor.hpp
│   ├── model.hpp
│   ├── tokenizer.hpp
│   ├── attention.hpp
│   └── runtime.hpp
├── src/
│   ├── main.cpp
│   ├── tensor.cpp
│   ├── model.cpp
│   ├── tokenizer.cpp
│   ├── attention.cpp
│   └── runtime.cpp
├── kernels/
│   ├── matmul_scalar.cpp
│   └── matmul_neon.cpp
├── tools/
│   └── export_model.py
└── models/
```

## 12. Reference inference implementation

Implement these operations first:

``` text
embedding lookup
matrix multiplication
vector addition
LayerNorm
softmax
ReLU/GELU
causal attention
residual addition
argmax/sampling
```

The first runtime does NOT need to be highly optimized.

Correctness comes first.

## 13. Numerical correctness strategy

This is important.

Before testing performance, compare the C++ runtime against PyTorch.

For the exact same input tokens:

``` text
PyTorch logits
      vs
C++ logits
```

Calculate:

``` text
max absolute error
mean absolute error
relative error
```

For FP32, these should be very small.

Then compare generated token sequences.

This prevents accidentally debugging "AI quality" when the actual
problem is a matrix multiplication bug.

## 14. Cross compilation

The existing embedded workflow is useful here.

Host:

``` text
x86_64 Linux / WSL
```

Target:

``` text
ARMv7
BeagleBone Black
```

Use the existing ARM cross-toolchain where practical.

Compile with:

``` text
-march=armv7-a
-mfpu=neon
-mfloat-abi=hard
```

The exact compiler flags must be verified against the BBB's existing
ABI/toolchain.

First confirm:

``` bash
gcc -v
gcc -dumpmachine
cat /proc/cpuinfo
ldd --version
```

on the target/toolchain.

## 15. NEON optimization

Only optimize after the scalar runtime is correct.

The first target should be matrix multiplication.

Conceptually:

``` text
C = A × B
```

Start:

``` text
for i
    for j
        for k
            C[i,j] += A[i,k] * B[k,j]
```

Then create a NEON implementation.

Potential progression:

``` text
scalar FP32
    ↓
cache-friendly tiling
    ↓
NEON FP32
    ↓
INT8 weights
    ↓
NEON INT8 / mixed precision
```

Benchmark each stage.

The goal is to learn where the time actually goes rather than optimizing
everything.

## 16. Memory budget

Treat RAM as a first-class constraint.

BBB:

``` text
512 MB total RAM
```

The OS and other services consume part of it.

Set an initial project target:

``` text
model       < 10 MB
runtime     < 20 MB RSS overhead
working RAM < 100 MB
```

These are engineering targets.

Measure actual RSS using:

``` bash
/usr/bin/time -v ./tiny_transformer ...
```

and:

``` bash
cat /proc/<PID>/status
```

Relevant values:

``` text
VmRSS
VmPeak
VmHWM
```

## 17. Benchmark suite

Every optimization should be measured.

Record:

``` text
Model:
Parameters:
Model file size:
Precision:
Context:
Prompt length:

Prompt processing:
  ms

Generation:
  ms/token
  tokens/sec

Memory:
  VmRSS
  VmHWM

CPU:
  utilization
```

Create:

``` text
benchmarks/
└── results.md
```

Example:

``` text
                 FP32 scalar    FP32 NEON    INT8
----------------------------------------------------
Model size          1.8 MB         1.8 MB      0.5 MB
RSS                 42 MB          42 MB       30 MB
ms/token             820            410         190
tokens/sec           1.2            2.4         5.3
```

The numbers above are placeholders. Measure the real values.

## 18. Optional embedded service

If time remains:

``` text
tiny-transformer.service
```

systemd service:

``` text
[Service]
ExecStart=/opt/tiny-transformer/tiny_transformer_server
User=tinyai
Restart=on-failure
```

Expose a tiny interface.

### Simplest

CLI:

``` bash
tiny-transformer "ROMEO:"
```

### More interesting

HTTP:

``` text
POST /generate
{
    "prompt": "ROMEO:",
    "max_tokens": 32
}
```

### Most aligned with existing experience

MQTT:

``` text
ai/request
ai/response
ai/status
```

## 19. Stretch goal: robotics/embedded AI angle

After the language model works, create a tiny command model.

Input:

``` text
move forward ten centimeters
turn left ninety degrees
stop
```

Output:

``` text
MOVE 10
TURN LEFT 90
STOP
```

Then the BBB becomes:

``` text
natural language
       ↓
tiny Transformer
       ↓
structured command
       ↓
validation
       ↓
MQTT / GPIO / robot interface
```

Important:

The Transformer must NOT directly control hardware.

Put a deterministic validator between the model and actuator.

``` text
AI
 ↓
parser
 ↓
schema validation
 ↓
limits / safety checks
 ↓
action
```

This turns the project into a much more realistic embedded-AI
architecture.

## 20. Weekend schedule

## Saturday morning --- Model

To train a tiny Transformer.

Tasks:

- [ ] Inspect BBB hardware/software
- [ ] Set up clean project repository
- [ ] Download/select tiny dataset
- [ ] Implement character tokenizer
- [ ] Implement tiny Transformer
- [ ] Implement training loop
- [ ] Overfit tiny sample
- [ ] Train full small model
- [ ] Generate sample text
- [ ] Save checkpoint

### Saturday afternoon --- Export

- [ ] Define model binary format
- [ ] Write exporter
- [ ] Export weights
- [ ] Write binary loader test
- [ ] Verify every tensor shape
- [ ] Verify checksums/sizes

### Saturday evening --- C++ runtime

- [ ] Implement tokenizer
- [ ] Implement tensor storage
- [ ] Implement matmul
- [ ] Implement LayerNorm
- [ ] Implement attention
- [ ] Implement MLP
- [ ] Implement autoregressive loop

### Saturday night milestone

Run the C++ implementation on the laptop and compare it with PyTorch.

## 21. Sunday schedule

## Sunday morning --- BBB

- [ ] Verify BBB architecture
- [ ] Verify compiler/toolchain
- [ ] Build ARMv7 binary
- [ ] Transfer binary
- [ ] Transfer model
- [ ] Run inference
- [ ] Fix ABI/library issues
- [ ] Verify output against laptop

### Sunday afternoon --- Performance

- [ ] Benchmark scalar implementation
- [ ] Profile hot functions
- [ ] Implement NEON matmul
- [ ] Benchmark again
- [ ] Optimize memory layout
- [ ] Try INT8 if time permits

### Sunday evening --- Make it a real embedded project

- [ ] Add benchmark report
- [ ] Add systemd service
- [ ] Add CLI/API
- [ ] Add model version metadata
- [ ] Document deployment
- [ ] Capture final demo

------------------------------------------------------------------------

## 22. Explicit decision points

## Choice A --- Character vs BPE

Recommended:

``` text
Weekend 1:
character-level
```

Reason:

The tokenizer is not where we want to spend the weekend.

Later:

``` text
BPE/token-level
```

## Choice B --- Python runtime vs C++

Recommended:

``` text
Training: Python/PyTorch
Inference: C++
```

This creates a clean separation between research and deployment.

## Choice C --- FP32 vs INT8

Recommended:

``` text
Phase 1: FP32
Phase 2: INT8
```

Do not start with quantization.

First prove numerical correctness.

## Choice D --- existing inference library vs own runtime

Recommended:

``` text
own minimal runtime
```

If the objective were merely "get a model running", use an existing
runtime.

The objective here is to learn the embedded inference stack.

## Choice E --- architecture

Recommended:

``` text
tiny decoder-only Transformer
```

Do not implement encoder-decoder, RoPE, GQA, MoE, FlashAttention, etc.
initially.

------------------------------------------------------------------------

## 23. Stretch architecture after the weekend

If the weekend succeeds, the project can evolve:

``` text
                     tiny-transformer
                           |
          +----------------+----------------+
          |                                 |
       Training                        Embedded runtime
          |                                 |
       PyTorch                         C++ / NEON
          |                                 |
     checkpoint                         model.bin
          |                                 |
        export                              |
          +---------------> BBB <-----------+
                              |
                    systemd inference daemon
                              |
                   +----------+----------+
                   |                     |
                  MQTT                 GPIO/CAN
```

Then experiment with:

- BPE tokenizer
- RoPE
- larger context
- INT8
- KV cache
- weight-only quantization
- memory-mapped model files
- streaming generation
- NEON kernels
- model hot-swapping
- OTA model deployment
- MQTT model management
- watchdog/restart
- inference telemetry

------------------------------------------------------------------------

## 24. What NOT to do this weekend

Avoid:

- Llama/Qwen/etc.
- Hugging Face dependency on the BBB
- installing PyTorch on the BBB
- building a huge tokenizer stack
- CUDA-specific code
- complex inference frameworks
- Kubernetes/container orchestration
- ROS integration before inference works
- premature quantization
- premature NEON optimization
- building a web UI

The core victory is:

> **I trained a Transformer myself, exported it, cross-compiled an
> inference runtime, and generated tokens on a 1 GHz ARMv7 embedded
> Linux board with 512 MB RAM.**

Everything else is bonus.

------------------------------------------------------------------------

## 25. Repository layout

Suggested final repository:

``` text
bbb-tiny-transformer/
│
├── README.md
├── PLAN.md
├── LICENSE
│
├── training/
│   ├── train.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── dataset.py
│   └── generate.py
│
├── export/
│   └── export_model.py
│
├── runtime/
│   ├── CMakeLists.txt
│   ├── include/
│   ├── src/
│   └── kernels/
│
├── models/
│   └── tiny-transformer.bin
│
├── benchmarks/
│   ├── results.md
│   └── scripts/
│
├── systemd/
│   └── tiny-transformer.service
│
└── docs/
    ├── architecture.md
    ├── model-format.md
    └── deployment.md
```
