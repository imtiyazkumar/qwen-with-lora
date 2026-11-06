# Phase 3: model selection and local inference

## Decision

Use **Qwen/Qwen3.5-2B**, pinned to revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, in its official BF16
Safetensors format. The [official license](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE)
is Apache 2.0; a copy is downloaded beside the model. No remote Python code
is enabled. Only this candidate's weights were downloaded.

The checkpoint is a vision-language model with hybrid Gated Delta / full
attention. For this text-document assistant, Transformers loads its language
component as `Qwen3_5ForCausalLM`, omitting vision weights from GPU memory.
All text parameters loaded without missing weights. This text-only class is
documented in the [official Transformers model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
The complete downloaded checkpoint still contains the vision weights.

## Candidate comparison

Official metadata and configurations were inspected for all three candidates.
The sizes below describe their published full checkpoints, not VRAM usage.

| Candidate | License | Published weights | Decision |
| --- | --- | --- | --- |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Apache 2.0 | 4.55 GB | Selected; tested BF16 text inference fits the GPU. |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Apache 2.0 | 1.75 GB | Fallback if later measured training memory requires it; not downloaded. |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Apache 2.0 | 8.04 GB | Standard transformer alternative; full BF16 weights exceed 6 GB. Quantized training would need separate validation. Not downloaded. |

Qwen3.5 needs Transformers v5, which is installed. Its architecture supports
LoRA through Unsloth, but [Unsloth recommends BF16 LoRA rather than 4-bit QLoRA](https://unsloth.ai/docs/models/qwen3.5/fine-tune).
The planned training configuration therefore retains BF16 base weights.
Actual training memory at rank 8, alpha 16, batch 1, sequence length 512,
gradient accumulation 8, checkpointing enabled, and one initial epoch must
still be measured in Phase 5. Inference fit does not establish training fit.

## Runtime comparison

| Runtime | Verified upstream support | Project decision |
| --- | --- | --- |
| Transformers with Unsloth's bundled kernels | Native Qwen3.5 Safetensors, text-only class, PEFT adapters | Selected and tested locally. Uses the same checkpoint format intended for training. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp) | Native Qwen3.5 implementation with GGUF and CUDA backend | Suitable deployment candidate after an explicit export and adapter compatibility test; not installed or benchmarked here. |
| [LM Studio](https://lmstudio.ai/models/qwen3.5) | Lists Qwen3.5-2B and GGUF support | Optional desktop runtime; not installed or benchmarked here. |
| [Ollama](https://ollama.com/library/qwen3.5) | Publishes explicit `qwen3.5:2b` and `0.8b` models | Optional local service; not installed or benchmarked here. |

This phase avoids downloading a second model format or introducing an
additional service. The selected runtime uses local Gated Delta kernels
bundled with Unsloth. The initial baseline below used Transformers' correct
PyTorch convolution fallback. The optional `causal_conv1d` extension was
subsequently installed and tested, as described below. Neither step required
installing a CUDA toolkit or compiling that extension locally.

## Measured results

The saved [Phase 3 report](../evaluation/reports/phase3-model.json) comes from
a fresh process using local files and offline settings on the RTX 3050 6 GB.

| Measurement | Result |
| --- | --- |
| Loaded language parameters | 1,881,825,088 |
| Precision / placement | BF16, entirely on `cuda:0`; no CPU or disk offload |
| Warm-file-cache weight loading | 1.70 seconds, excluding Python/framework imports |
| Longer response benchmark | 128 generated tokens in 6.92 seconds |
| Decode throughput after first token | 18.61 tokens/second |
| Throughput including prefill | 18.50 tokens/second |
| Benchmark time to first token | 0.094 seconds after warmup |
| Larger-context probe | 567 input tokens; returned the correct supplied time |
| Largest measured PyTorch allocation | 3,670.4 MiB, approximately 3.58 GiB |
| Total GPU usage after that probe | 4,728.5 MiB, approximately 4.62 GiB |
| Remaining VRAM after that probe | 1,415 MiB |
| Basic conversation checks | 5/5 passed |
| Checkpoint SHA256 and sizes | Passed |
| Python network connection attempts | Zero |

GPU totals include the CUDA context and other device allocations; PyTorch
allocation counters do not. Total usage is a snapshot after generation,
whereas the PyTorch allocation figure is a peak. These results describe the
tested short contexts, not the model's advertised 262K context window.
Do not load a second GPU model concurrently on this laptop.

The five checks cover a simple fact, arithmetic, explicitly supplied context,
missing information, and multi-turn memory. They are smoke tests, not a broad
accuracy or hallucination benchmark. The longer response deliberately reaches
the configured 128-token output cap and is marked truncated in the report.

## Native convolution acceleration

The official [Dao-AILab v1.7.0 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)
provides a wheel matching this environment: CPython 3.12, Linux x86_64,
PyTorch 2.10, CUDA 12, and CXX11 ABI=true. That wheel and its required Ninja
package are now installed. The release's SHA256 is pinned in
`config/requirements-causal-conv.txt`; the complete environment lock includes
both packages. Existing PyTorch and driver versions were retained.

`scripts/verify_causal_conv.py` compares forward operations, backward
gradients, and incremental decoding-state updates against PyTorch in FP32
and BF16. It also checks that a fresh Transformers process selects the native
implementations for both functions named in the fallback warnings. Results
are in `evaluation/reports/phase3-causal-conv.json`.

An already-running chat retains its original function bindings. Enter
`/quit`, then restart `scripts/chat_local.py` to activate native convolution.
After the running chat was closed, the full model was tested again in a fresh
offline process. Both reports are retained; the table above is the original
fallback baseline.

### Accelerated model benchmark

The same model revision, prompts, token limits, BF16 precision, and GPU were
used. The [accelerated report](../evaluation/reports/phase3-model-accelerated.json)
and [comparison](../evaluation/reports/phase3-acceleration-comparison.json)
record these results:

| Measurement | Fallback baseline | Native convolution |
| --- | --- | --- |
| Generated benchmark tokens | 128 | 128 |
| Decode tokens/second | 18.61 | 19.64 |
| Generation time including prefill | 6.921 s | 6.532 s |
| Benchmark time to first token | 0.094 s | 0.064 s |
| GPU usage after 567-token probe | 4,728.5 MiB | 4,728.5 MiB |
| Conversation smoke checks | 5/5 | 5/5 |

The long-response decode rate was approximately **5.5% higher in this run**.
This is one warm run per environment; short-response and context timings
varied, so it is not a controlled multi-run speedup estimate. The generated
benchmark text and all five smoke-test answers were identical. Offline checks,
the larger-context probe, and input limits passed again. No convolution
fallback warnings appeared in the accelerated run.

Reproduce the accelerated run while the GPU is free:

```bash
.venv/bin/python scripts/verify_model.py \
  --output evaluation/reports/phase3-model-accelerated.json
```

## Offline behavior and limits

The tokenizer, chat template, configuration, license, and checkpoint live in
`models/base/Qwen3.5-2B`. `config/model-source.json` records the upstream
revision, file sizes, and checkpoint SHA256. `config/model.json` controls the
local runtime. Loading uses `local_files_only=True` and
`trust_remote_code=False`.

Offline verification blocks Python socket connections and DNS lookups,
sets Hugging Face offline flags, disables optional telemetry, and loads and
generates in a fresh process. There were no attempted connections. This does
not claim that the Windows network adapter was disconnected or that native
extensions were isolated by an operating-system firewall.

The conservative application limits are 768 input tokens, 128 new tokens,
and a 1,024-token total context cap. Over-budget input raises an error rather
than silently dropping context. Thinking is disabled through the official
chat template. The terminal chat keeps history in memory and supports
`/clear` and `/quit`; it does not yet retrieve documents or create citations.

## Commands

From the project root in WSL:

```bash
# Explicit online setup step; reuses existing downloads at the pinned revision.
.venv/bin/python scripts/download_model.py

# Offline loading, integrity checks, conversations, and benchmarks.
.venv/bin/python scripts/verify_model.py

# Interactive local terminal chat.
.venv/bin/python scripts/chat_local.py
```

The scripts set their own process-local cache and privacy settings. Initial
Triton compilation may take longer on a fresh installation. Run in a normal
WSL terminal if an agent sandbox blocks GPU device access.
