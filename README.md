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
