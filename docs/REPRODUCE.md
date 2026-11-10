# Reproducing Inferno's results (for someone outside the project)

Goal: confirm, on your own hardware, (1) that the headline figure regenerates from the published data and
(2) that the short GPU benchmark produces numbers of the same shape. Budget: ~30 min setup, ~5 min of GPU time.

## What you need

- Linux or WSL2 with an NVIDIA GPU (Ampere or newer: RTX 30/40, A10G, A100, L4, H100), driver supporting CUDA 12.x.
- ~15 GB free disk (Python env + the 1.5B AWQ model download).

## Steps

```bash
git clone https://github.com/Likhith1807/Inferno-Adaptive-LLM-Inference-Serving.git inferno && cd inferno
git checkout v1.0.0                       # the tagged release this guide was written against

# 1. No GPU: headline figure from the saved data
python -m pip install matplotlib
make figure                               # -> docs/figures/headline_light.png; compare with the README image

# 2. Environment (date-pinned so it resolves to CUDA 12.x builds)
bash scripts/setup_env.sh
bash scripts/setup_toolchain.sh           # nvcc, gcc (no root needed)
source scripts/toolchain_env.sh
make test                                 # unit tests, no GPU

# 3. Short GPU benchmark
make quick-bench                          # -> results/quick/<timestamp>/
```

On a non-Ada GPU, build the kernel for your architecture first: `make -C src/inferno/kernels ARCH=sm_86 libinferno_int4.so`
(sm_80 A100, sm_86 RTX 30 / A10G, sm_89 RTX 40 / L4, sm_90 H100).

## What to send back

Open an issue titled "Reproduction: <your GPU>" and attach the three files from `results/quick/<timestamp>/`:

| file | what it shows |
|---|---|
| `hardware.txt` | GPU, driver, clocks, torch / CUDA / vLLM versions (identifies the hardware) |
| `layer_matrix.json` | custom vs Marlin vs AWQ vs fp16 on the qkv and gate_up layers, batch 1 and 16 |
| `marlin_probe.txt` | whether Marlin returns wrong results at batch 32/64/128 on your machine |

Also say whether step 1's figure matched, and anything in this guide that was unclear or wrong.

## What to expect

- Batch 1: custom and Marlin within ~10% of each other; batch 16: Marlin several times faster than custom.
- Absolute microseconds scale with your GPU's memory bandwidth (the RTX 4060 Laptop is 256 GB/s).
- `marlin_probe.txt`: on the original machine (RTX 4060 Laptop, WSL2, vLLM 0.11.0) batches 32/64/128 show bad calls.
  **Either outcome on your machine is useful**: zero bad calls suggests the fault is specific to that machine.

## Reproductions so far

| date | who | GPU / OS | figure matched | quick-bench | Marlin probe |
|---|---|---|---|---|---|
| (none yet) | | | | | |
