# Informative Drawings (standalone ZML example)

> NOTE: This project is vibe-coded.

This is a standalone Bazel project that uses [ZML](https://github.com/zml/zml) as an external dependency to run the [Informative Drawings](https://github.com/carolineec/informative-drawings) model.

It converts photographs into line drawings using a neural style transfer model.

This repository also includes a (vibe-coded) Zig QOI encoder / decoder.

## Prerequisites

- Bazel - see [the ZML repo](https://github.com/zml/zml#prerequisites)
- [uv](https://docs.astral.sh/uv/) for weights conversion

## Model weights

Download the PyTorch weights and convert them to safetensors:

```bash
./convert.py
```

## Build and run

```bash
bazel build --config=release //:informativedrawings

bazel run --config=release //:informativedrawings -- \
    "$PWD"/model2.safetensors "$PWD"/zml.qoi "$PWD"/zml.drawing.qoi
```
