[中文](README-zh.md)

# What is GPTVM

GPTVM, as the name suggests, is a Virtual Machine architecture for Generative
Pre-trained Transformer models. GPTVM aims to provide a systematic optimization
and execution infrastructure to support primarily large language model serving
applications running on various target platforms from single GPU to large-scale
distributed heterogeneous systems.

As a virtual machine, GPTVM provides an abstract layer filling the gap between
front-end language model generation APIs and back-end model implementations.
Front-end LLM frameworks (vLLM, TGI, DeepSpeed, ...) usually focus on high-level
python level optimizations, such as introducing KV cache to avoid redundant
computation, re-organizing inference requests to saturate large batches for
better GPU utilization (continuous batching), parallel execution across multiple
accelerator devices (TP, PP). In low-level backends, optimizations usually take
place at model or operator level, including fused operators (flash attention),
quantizations, etc. This two-end solution is not a systematic approach for optimizing
LLM applications, thus has the following drawbacks: 1) It requires a great amount
of hand-crafted coding effort for each model. 2) It lacks the extensibility of
implementing new optimizations. 3) It is difficult to ensure the scalability of
deploying applications on a larger scale heterogeneous distributed system.

To address these, we feel the necessity of introducing a mid-level program
abstraction for LLM applications to enable previously mentioned LLM optimizations
in a compiler-native approach similar to the traditional language virtual machine
architecture, so that we can bring all traditional language virtual machine
benefits to LLM applications:

* Platform-independent - compile once, run everywhere, including on large-scale
  distributed heterogeneous systems
* Model-independent - generic infrastructure without model-specific optimization
  code
* Compiler lowering and optimizations - transform high-level generation APIs to
  LLM primitives in GPTVM, alongside whole program static optimizations
* Managed runtime - allows fine control of program execution at runtime and provides
  a managed memory abstraction for programs running on the VM, which is crucial for
  LLM applications for its extensive memory requirement.

# Key Features

* Multi-backend support

  GPTVM provides a platform-independent VM architecture for LLM applications to
  support various kinds of accelerators as VM backends. The VM defines a set of
  LLM primitives left for backend implementation. This enables easy deployment of
  LLM applications to non-NV GPUs and other kinds of accelerators.

* Targeting large-scale distributed heterogeneous platforms

  GPTVM is designed to target large-scale distributed systems. GPTVM runtime employs
  an actor-based execution environment which ensures the scalability of LLM
  applications in a distributed system.

* Automatic compiler and runtime optimizations to achieve the state-of-the-art
  performance of LLM serving

  The optimization in GPTVM is designed to be effective for all LLM models and
  applications. With a well-designed abstraction layer provided by GPTVM, the
  compiler and the runtime could optimize the program in a model-independent and
  platform-independent way.

* Automatic tensor/pipeline parallel

  GPTVM can automatically select the best parallel strategy for the given LLM
  model input and the runtime platform specification. GPTVM compiler analyzes
  the computation and memory access pattern from the model, then decomposites
  the large model into sub-tasks and makes sure each sub-task fits the computation
  power and memory resource in a single backend device.  GPTVM runtime is
  responsible for scheduling sub-tasks on devices and managing the memory allocation
  and data transfer between devices.

# Build

## Prerequisites

The GPTVM build system is designed to be as independent as possible on
pre-installed packages, except basic development tools and some packages
required by third-party components. Following is the list of required packages
on Ubuntu 22.04.

```shell
apt install build-essential python3 python3-dev cmake protobuf-compiler libjsoncpp-dev libjsoncpp25
```

## Update submodules

Make sure all submodules are updated.

GPTVM only depends on its direct submodules as specified in .gitmodules, no
need for a recursive update.

```shell
git submodule update --init
```

## cmake build

Simply create a build directory under source directory and cmake it.

```shell
cmake .. && make
```
or
```shell
cmake -G Ninja .. && ninja
```
if you want to use Ninja to build.

## Install

To build a release packge, append `-DCMAKE_INSTALL_PREFIX=<install_dir>` to
your cmake command line options, and run `make install` or `ninja install`.

# Usage

Export `PYTHONPATH` to include both build and source directories.
```shell
export PYTHONPATH=$PYTHONPATH:$PWD/gptvm/python:$PWD/../python
```

Export `PATH` to include the `gptvm` command line driver.
```shell
export PATH=$PATH:$PWD/bin
```

Invoke `gptvm` instead of `python` then all the magic happens.
```shell
gptvm <your_application.py>
```
