# Local AI assistant: environment setup

## Selected environment

Use the existing **Ubuntu 26.04 LTS on WSL2**, as requested. Do not install
Ubuntu 24.04. Use the existing project `.venv`, running Python 3.12.16;
Ubuntu's system Python 3.14 is not the project interpreter.

Read-only checks on 2026-11-03 confirmed:

- Windows 11 Home Single Language, 64-bit; Ryzen 7 7840HS.
- NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM; Windows NVIDIA driver 592.82.
- WSL 2.7.10.0; the registered `Ubuntu` distribution uses WSL2.
- CUDA driver initialization succeeds; GPU compute capability is 8.6.
- WSL sees approximately 7.4 GiB RAM and 2 GiB swap. Windows reported
  approximately 15.3 GiB usable RAM and 1.7 GiB free at inspection time.
- Windows C: reported approximately 485 GiB free. The larger free-space
  figure inside WSL reflects its virtual filesystem, not extra physical space.

No Windows settings or NVIDIA drivers were changed. Do not install an NVIDIA
Linux display driver inside WSL. The CUDA version printed by `nvidia-smi`
describes driver capability; the PyTorch build below uses CUDA 12.8.

## PyTorch installation and Phase 1 verification

**Status: passed.** PyTorch 2.10.0+cu128 executed the GPU forward and backward
checks successfully. NumPy 2.5.4 and Triton 3.6.0 imports passed, and `uv pip
check` found no dependency conflicts across the 29 installed packages.
The saved hardware test is `evaluation/reports/phase1-gpu.json`.

Run from the project root in Ubuntu. `uv` is already installed. This command
installs into the existing project environment, not system Python, and uses
the official PyTorch CUDA 12.8 wheel index through uv's backend selector.
GPU runtime dependencies are large; this step does not download model weights.

```bash
uv --no-config --cache-dir .cache/uv pip install \
  --python .venv/bin/python \
  --default-index https://pypi.org/simple \
  --torch-backend cu128 \
  --only-binary :all: \
  -r config/requirements-gpu.txt

.venv/bin/python scripts/verify_gpu.py \
  --output evaluation/reports/phase1-gpu.json

uv --no-config --cache-dir .cache/uv pip check --python .venv/bin/python
```

The exact installed package versions are saved in
`config/requirements-gpu.lock.txt`. To reproduce this baseline in an existing
Python 3.12 virtual environment, use the same installation command above with
`-r config/requirements-gpu.lock.txt` instead of the shorter requirements file.

The verification performs an actual GPU matrix multiplication and backward
pass, checks both results, and reports GPU memory. Run it from an ordinary WSL
terminal if an agent sandbox blocks access to `/dev/dxg`. Passing this small
test does not establish that model inference or fine-tuning will fit in VRAM.

## Phase 2: training tools and Jupyter

**Status: passed.** The following versions are installed in the same Python
3.12 environment. The dependency checker passes for all 174 packages.

| Component | Tested version |
| --- | --- |
| Unsloth / Unsloth Zoo | 2026.10.11 |
| Transformers | 5.17.0 |
| PEFT | 0.21.2 |
| TRL | 1.15.0 |
| bitsandbytes | 0.50.2 |
| Accelerate | 1.15.0 |
| Datasets | 4.8.5 |
| xformers | 0.0.34 |
| TorchAO | 0.16.0+cu128 |
| Jupyter Notebook | 7.6.4 |
| IPython kernel | 7.4.0 |

TorchAO is explicitly pinned: 0.17's native extensions require PyTorch 2.11
or newer, while [TorchAO 0.16's compatibility table](https://github.com/pytorch/ao/blob/v0.16.0/torchao/__init__.py)
includes PyTorch 2.10. The GPU baseline remains PyTorch 2.10.0+cu128.

### Reproduce the development environment

The direct requirements are in `config/requirements-training.txt`. To reproduce
the exact tested environment, install its full version lock instead:

```bash
uv --no-config --cache-dir .cache/uv pip install \
  --python .venv/bin/python \
  --default-index https://pypi.org/simple \
  --torch-backend cu128 \
  --only-binary :all: \
  -r config/requirements-training.lock.txt

source scripts/local_env.sh

.venv/bin/python -m ipykernel install \
  --prefix "$PWD/.venv" \
  --name local-ai \
  --display-name 'Local AI (Python 3.12)'

.venv/bin/python scripts/verify_training.py \
  --output evaluation/reports/phase2-training.json

uv --no-config --cache-dir .cache/uv pip check --python .venv/bin/python
```

The `local-ai` kernel is already registered in this project's virtual
environment. Registration is required again if the environment is recreated
or moved, because its kernel specification contains an absolute Python path.

### What was tested

`evaluation/reports/phase2-training.json` records successful checks for:

- Imports of the training libraries, quantization libraries, and Jupyter.
- bitsandbytes NF4 packed weights with a GPU forward and backward pass.
- xformers attention against PyTorch SDPA, with finite gradients.
- An actual Unsloth Triton RMSNorm kernel against a PyTorch reference,
  including gradients.
- One TRL SFT step with PEFT LoRA on a tiny randomly initialized Llama model.
  Adapter parameters changed and frozen base parameters remained identical.
- Adapter saving, exact parameter reloading, output comparison, disabling,
  and unloading. Output comparisons use matching mixed precision.
- A real Jupyter `local-ai` kernel executing a GPU calculation with the
  project's Python interpreter.

The synthetic training test uses explicit zero warmup and a constant learning
rate, so its single step updates the adapter. Unsloth otherwise defaults to
fractional warmup. It uses BF16 mixed precision on this GPU. These are smoke
test settings, not the production training configuration.

The framework checks use offline environment settings and block Python socket
connections and DNS lookups; none were attempted. Jupyter's separate kernel
test permits local sockets. This is not a whole-system network isolation test.
No pretrained weights or private documents were used. Actual model memory
requirements and model-specific fine-tuning compatibility remain unverified.

### Launch Jupyter locally

```bash
bash scripts/start_notebook.sh
```

Open the local URL printed in that terminal in your Windows browser and select
**Local AI (Python 3.12)** as the notebook kernel. The server binds to
`127.0.0.1:8888`, keeps Jupyter's token authentication, and serves this project
directory. Stop it with Ctrl+C in its terminal.

The launcher sources `scripts/local_env.sh`, which keeps Hugging Face,
Triton, compiler, Jupyter, and IPython caches inside the project `.cache`
directory. It defaults to offline mode and disables optional Hugging Face
telemetry, Unsloth statistics, and Weights & Biases tracking. Source this
environment before running training code as well. Import Unsloth before
Transformers, PEFT, or TRL in training notebooks.

## Phase 3: selected model and local inference

**Status: passed.** The official Apache-2.0 Qwen3.5-2B checkpoint is downloaded
at a pinned revision. Its BF16 text model loaded entirely onto the RTX 3050.
Checkpoint integrity, tokenizer/configuration loading, five conversation
checks, a 567-token context probe, and offline verification passed.

The initial generation benchmark measured **18.61 decode tokens/second**.
After native convolution was installed, the same benchmark measured
**19.64 decode tokens/second**, approximately 5.5% higher in that run.
Total GPU usage after the larger-context probe was approximately **4.62 GiB**;
peak PyTorch tensor allocation was approximately **3.58 GiB**. These figures
have different scopes, as explained in the report.

```bash
.venv/bin/python scripts/chat_local.py
```

Use `/clear` to reset history or `/quit` to exit. The chat currently uses model
knowledge and any context you type. It has no document retrieval connected.
Responses are capped at 128 new tokens; chat input is capped at 768 tokens.

See [model selection and measured limits](model-selection.md),
the [runtime configuration](../config/model.json), and
the [Phase 3 report](../evaluation/reports/phase3-model.json).
Re-run verification with:

```bash
.venv/bin/python scripts/verify_model.py
```

No fallback model or GGUF copy was downloaded. No new Python packages were
needed for the initial Phase 3 baseline. Native causal-convolution acceleration
was subsequently installed and verified as described below. Model-specific
BF16 LoRA training fit remains to be measured in Phase 5.

### Causal-convolution fallback messages

These messages indicate that the model is using a correct, slower PyTorch
implementation. The matching official `causal-conv1d` 1.7.0 wheel and required
Ninja 1.13.2 package are now installed. No CUDA toolkit or display driver was
installed. The full environment lock now records 176 packages, including this
extension and Ninja; the earlier Phase 2 report records its original snapshot.

For this exact Python/PyTorch environment, the acceleration can be reproduced
without rebuilding or changing PyTorch:

```bash
uv --no-config --cache-dir .cache/uv pip install \
  --python .venv/bin/python \
  --only-binary :all: --no-deps \
  -r config/requirements-causal-conv.txt

.venv/bin/python scripts/verify_causal_conv.py
```

The [native-kernel report](../evaluation/reports/phase3-causal-conv.json)
records GPU forward, backward, and decoding-state checks in FP32 and BF16,
plus native-function selection by a fresh Transformers process. The complete
model was also retested offline after the running chat was closed: all five
conversation checks passed, answers were identical, and the fallback warnings
were absent. Its longer-response benchmark measured 19.64 decode tokens/second
versus the initial 18.61. The 567-token probe still used approximately 4.62 GiB
total GPU memory. These are individual runs, not a multi-run speedup estimate.
See the [accelerated report](../evaluation/reports/phase3-model-accelerated.json)
and [comparison](../evaluation/reports/phase3-acceleration-comparison.json).

To activate it, enter `/quit` in the current chat and restart:

```bash
.venv/bin/python scripts/chat_local.py
```

## Next phase

Phase 4 will implement document ingestion, local embeddings, Chroma indexing,
updates/deletion, duplicate detection, retrieval, and source metadata.

References: [PyTorch installation](https://pytorch.org/get-started/locally/),
[Unsloth Core installation](https://unsloth.ai/docs/get-started/install/pip-install),
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
