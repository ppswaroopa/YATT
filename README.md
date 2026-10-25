# YATT - Yet Another Tiny Transformer

A tiny decoder-only Transformer built from scratch and deployed a custom
native inference runtime to a 1 GHz ARMv7 BeagleBone Black, an effort to
progressively optimize it for memory, latency, and NEON SIMD
execution.

## Goal

``` text
Laptop:

train.py
   ↓
tiny_transformer.pt
   ↓
export_model.py
   ↓
tiny_transformer.bin
   ↓
cross compile
   ↓
tiny_transformer_armv7
   ↓
scp
   ↓

BeagleBone Black:

$ ./tiny_transformer tiny_transformer.bin "ROMEO:"

ROMEO: ...
...
tokens/sec: X
RSS: XX MB
```

Then:

``` text
$ ./tiny_transformer --benchmark

FP32 scalar
    X ms/token

FP32 NEON
    Y ms/token

speedup: Zx
```

## Build

Follow build.md for build instructions

### ARMv7 Performance Baseline

```text
========================================
YATT Baseline Benchmark
========================================
Model:              yatt_model.bin
Prompt:             First Citizen:
Prompt tokens:      14
Generated tokens:   100
Context length:     64
Embed dimension:    64
FFN dimension:      256
Attention heads:    4

Timing:
  Model load:       2.974 ms
  Tokenization:     0.004 ms
  Generation:       10144.472 ms
  Total:            10147.450 ms

Generation:
  Avg latency:      101.445 ms/token
  Throughput:       9.858 tokens/sec

Memory:
  Process RSS:      2.340 MB

Output:
First Citizen:
A Forir for main ond danathous and how athe ong holt that the of shen theerert suld for sitrald ry
========================================
```

### x86 Performance Baseline

```text
========================================
YATT Baseline Benchmark
========================================
Model:              checkpoints/yatt_model.bin
Prompt:             First Citizen:
Prompt tokens:      14
Generated tokens:   100
Context length:     64
Embed dimension:    64
FFN dimension:      256
Attention heads:    4

Timing:
  Model load:       4.563 ms
  Tokenization:     0.002 ms
  Generation:       1640.261 ms
  Total:            1644.826 ms

Generation:
  Avg latency:      16.403 ms/token
  Throughput:       60.966 tokens/sec

Memory:
  Process RSS:      4.125 MB

Output:
First Citizen: whevear yous to ben be the comend sour th nomarsouel, erast word autrin forne.

Miten, yos heay.

T
========================================
```
