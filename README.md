# Private local AI knowledge assistant

This project will combine document retrieval with a local open-weight model.
RAG will supply document facts and references; optional LoRA adapters will
customize model behavior. Documents and inference will remain local.

For terminology, concepts, process diagrams, and a study roadmap, see the
[project learning guide](docs/learning-guide.md).

## Current progress

Phases 1–3 are verified on the existing **Ubuntu 26.04 LTS / WSL2** installation.
Ubuntu 24.04 will not be installed.

- Python 3.12.16 in the project `.venv`.
- PyTorch 2.10.0 with CUDA 12.8 installed and tested on the RTX 3050 6 GB.
- Actual GPU matrix multiplication and backward pass both passed.
- NumPy and Triton imports passed; dependency consistency checks passed.
- Unsloth, Transformers, PEFT, TRL, and Jupyter are installed and tested.
- GPU NF4, attention, and Unsloth kernel tests passed.
- A synthetic TRL/LoRA update, frozen-base check, and adapter round trip passed.
- Jupyter's project kernel executed a GPU calculation successfully.
- The official Qwen3.5-2B model is downloaded and verified for offline use.
- BF16 text inference fits entirely on the GPU: approximately 4.62 GiB total
  GPU usage after the 567-token context probe; approximately 19.6 decode tokens/s
  in the latest 128-token benchmark with native convolution.
- Five conversation smoke checks passed. Actual model training is pending.

See [setup instructions](docs/setup.md), the
[GPU verification report](evaluation/reports/phase1-gpu.json),
[training-stack verification report](evaluation/reports/phase2-training.json),
and [full installed package versions](config/requirements-training.lock.txt).

Start an offline terminal chat:

```bash
.venv/bin/python scripts/chat_local.py
```

Use `/clear` or `/quit`. This is general model chat; document retrieval and
source citations will be added in Phase 4. Responses are capped at 128 tokens.
See [model selection and benchmarks](docs/model-selection.md) and the
[Phase 3 report](evaluation/reports/phase3-model.json).

Native `causal-conv1d` acceleration is also installed and
[verified](evaluation/reports/phase3-causal-conv.json). Restart an existing
chat to activate it. The [full accelerated benchmark](evaluation/reports/phase3-model-accelerated.json)
passed offline without fallback warnings. Its long-response decode rate was
19.64 tokens/s versus the original 18.61; this is one run per environment.

Start Jupyter on localhost:

```bash
bash scripts/start_notebook.sh
```

The next step is Phase 4: add the document knowledge base and retrieval.
Actual Qwen3.5-2B BF16 LoRA memory use remains to be measured in Phase 5.
# qwen-with-lora
