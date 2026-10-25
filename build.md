# Build

YATT currently supports two build targets:

- **Native build** — builds the C++ inference runtime for the development machine.
- **ARMv7 build** — cross-compiles the inference runtime for the BeagleBone Black.

The Python training/export workflow is independent of the C++ build.

## Prerequisites

For the native build:

- CMake
- C++ compiler with C++17 support

For the BeagleBone Black build:

- ARMv7 hard-float cross-toolchain
- CMake
- A BeagleBone Black running a compatible ARMHF Linux userspace

The current ARM toolchain used by the project is a Buildroot GCC 7.3.0 toolchain:

```text
arm-buildroot-linux-gnueabihf
```

### Native Build

From the repository root:

```bash
cmake -S deployment \
      -B deployment/build \
      -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build deployment/build -j$(nproc)
```

The resulting executable is:

```text
deployment/build/yatt_loader
```

Run it with an exported model:

```bash
./deployment/build/yatt_loader checkpoints/yatt_model.bin
```

### BeagleBone Black / ARMv7 Build

The ARM cross-compilation environment is configured using:

```text
toolchains/setup-bbb-toolchain.sh
```

Source the environment before configuring the build:

> Use this [repo](https://github.com/ppswaroopa/thinedge-beaglebone-lab/tree/main/toolchains) link for downloading the toolchain

```bash
#!/usr/bin/env bash

set -e

TOOLCHAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/armv7-eabihf--glibc--stable-2018.11-1" && pwd)"

export TOOLCHAIN_DIR
export SYSROOT="$TOOLCHAIN_DIR/arm-buildroot-linux-gnueabihf/sysroot"

export PATH="$TOOLCHAIN_DIR/bin:$PATH"

export CC="$TOOLCHAIN_DIR/bin/arm-buildroot-linux-gnueabihf-gcc"
export CXX="$TOOLCHAIN_DIR/bin/arm-buildroot-linux-gnueabihf-g++"

echo "BBB ARMv7 toolchain environment configured."
echo
echo "TOOLCHAIN_DIR=$TOOLCHAIN_DIR"
echo "SYSROOT=$SYSROOT"
echo "CC=$CC"
echo "CXX=$CXX"
echo
echo "Compiler:"
"$CXX" --version | head -1
```

Run:

```bash
source toolchains/setup-bbb-toolchain.sh
```

This configures:

```text
TOOLCHAIN_DIR
SYSROOT
CC
CXX
```

Verify the compiler:

```bash
echo "$CC"
echo "$CXX"
"$CXX" --version
```

Configure a separate ARM build directory:

```bash
cmake -S deployment \
      -B deployment/build-bbb \
      -DCMAKE_TOOLCHAIN_FILE=/home/pswaroopa/thinedge-beaglebone-lab/toolchains/bbb-armhf.cmake \
      -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build deployment/build-bbb -j$(nproc)
```

The resulting executable is:

```text
deployment/build-bbb/yatt_loader
```

Verify that the executable is an ARM hard-float binary:

```bash
file deployment/build-bbb/yatt_loader
```

Expected output includes:

```text
ELF 32-bit LSB executable, ARM, EABI5
```

You can also verify the ABI and dynamic linker:

```bash
readelf -h deployment/build-bbb/yatt_loader | \
    grep -E 'Class|Machine|Flags'

readelf -l deployment/build-bbb/yatt_loader | \
    grep interpreter
```

The expected interpreter is:

```text
/lib/ld-linux-armhf.so.3
```

### Deploying to the BeagleBone Black

Copy the ARM executable and exported model to the BeagleBone Black:

```bash
scp deployment/build-bbb/yatt_loader debian@<bbb-ip>:/tmp/
scp checkpoints/yatt_model.bin debian@<bbb-ip>:/tmp/
```

On the BeagleBone Black:

```bash
chmod +x /tmp/yatt_loader
cd /tmp
./yatt_loader yatt_model.bin
```

The executable is dynamically linked against the target's existing ARMHF userspace libraries. The cross-compilation toolchain itself does not need to be installed on the BeagleBone Black.

### Build Directories

Native and ARM builds use separate CMake build directories:

```text
deployment/
├── build/              # Native build
└── build-bbb/          # ARMv7 / BeagleBone Black build
```
