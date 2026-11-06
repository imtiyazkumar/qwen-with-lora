"""Test the local training stack on synthetic data without model downloads."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import socket
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]


def configure_environment() -> None:
    """Keep caches local and disable downloads and optional telemetry."""
    values = {
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"),
        "TRITON_CACHE_DIR": str(ROOT / ".cache/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache/torchinductor"),
        "UNSLOTH_COMPILE_LOCATION": str(
            ROOT / ".cache/unsloth_compiled_cache"
        ),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "JUPYTER_CONFIG_DIR": str(ROOT / ".cache/jupyter/config"),
        "JUPYTER_RUNTIME_DIR": str(ROOT / ".cache/jupyter/runtime"),
        "IPYTHONDIR": str(ROOT / ".cache/ipython"),
    }
    os.environ.update(values)


@contextmanager
def block_python_network() -> Iterator[list[str]]:
    """Block and record Python connections during framework tests."""
    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_lookup = socket.getaddrinfo

    def blocked(*args: Any, **kwargs: Any) -> Any:
        attempts.append("Python socket connection or DNS lookup")
        raise OSError(
            "Network disabled during the training stack verification"
        )

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.getaddrinfo = blocked
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.getaddrinfo = original_lookup


def check_imports() -> dict[str, str]:
    # Unsloth must be imported before Transformers, PEFT, and TRL.
    packages = (
        "unsloth",
        "unsloth_zoo",
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "xformers",
        "torchao",
        "triton",
        "torchvision",
        "notebook",
        "ipykernel",
        "pip",
    )
    versions = {}
    for name in packages:
        importlib.import_module(name)
        versions[name] = importlib.metadata.version(name)
    import torchao

    if torchao.skip_loading_so_files:
        raise RuntimeError("TorchAO skipped its native extensions")
    from trl import SFTConfig, SFTTrainer  # noqa: F401

    return versions


def check_four_bit() -> dict[str, Any]:
    import bitsandbytes as bnb
    import torch

    layer = bnb.nn.Linear4bit(
        64, 32, bias=False, compute_dtype=torch.float16, quant_type="nf4"
    ).cuda()
    x = torch.randn(
        4, 64, device="cuda", dtype=torch.float16, requires_grad=True
    )
    y = layer(x)
    y.float().square().mean().backward()
    if not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
        raise RuntimeError("NF4 forward/backward produced non-finite values")
    if layer.weight.dtype != torch.uint8:
        raise RuntimeError(
            "NF4 layer weights were not packed into 4-bit storage"
        )
    return {
        "status": "passed",
        "format": "nf4",
        "packed_dtype": str(layer.weight.dtype),
    }


def check_attention() -> dict[str, str]:
    import torch
    from xformers.ops import memory_efficient_attention

    shape = (1, 32, 2, 32)
    q, k, v = [
        torch.randn(
            shape, device="cuda", dtype=torch.float16, requires_grad=True
        )
        for _ in range(3)
    ]
    actual = memory_efficient_attention(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.003)
    actual.float().square().mean().backward()
    if not all(
        t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v)
    ):
        raise RuntimeError("Attention gradients failed")
    return {"status": "passed", "reference": "PyTorch SDPA"}


def check_unsloth_kernel() -> dict[str, str]:
    import torch
    from unsloth.kernels.rms_layernorm import fast_rms_layernorm

    layer = torch.nn.RMSNorm(32, eps=1e-6).cuda()
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    actual = fast_rms_layernorm(layer, x)
    expected = (
        x
        * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        * layer.weight
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual_grad = torch.autograd.grad(
        actual.square().sum(), x, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(
        actual_grad, expected_grad, rtol=0.01, atol=2e-5
    )
    return {"status": "passed", "kernel": "Triton RMSNorm forward/backward"}


def check_lora(directory: Path) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )
    from trl import SFTConfig, SFTTrainer

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=False,
    )
    base = LlamaForCausalLM(config).cuda()
    initial_base = {
        name: value.detach().clone()
        for name, value in base.state_dict().items()
    }
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    frozen = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    vocabulary.update({f"token{i}": i for i in range(4, 64)})
    tokenizer_impl = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    dataset = Dataset.from_dict(
        {"text": ["token4 token5 token6 [EOS]", "token7 token8 token9 [EOS]"]}
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    evaluation_dtype = torch.bfloat16 if use_bf16 else torch.float16
    args = SFTConfig(
        output_dir=str(directory / "trainer"),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_length=32,
        learning_rate=1e-3,
        # Fractional warmup would consume the entire one-step test.
        warmup_steps=0,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=False,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataset_num_proc=1,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    training_result = trainer.train()
    if training_result.global_step != 1 or not math.isfinite(
        training_result.training_loss
    ):
        raise RuntimeError("Synthetic TRL training step failed")
    current = dict(model.named_parameters())
    if not any(
        not torch.equal(current[name], value)
        for name, value in trainable.items()
    ):
        raise RuntimeError("No LoRA parameters changed during training")
    if not all(
        torch.equal(current[name], value) for name, value in frozen.items()
    ):
        raise RuntimeError(
            "Frozen base parameters changed during LoRA training"
        )
    model.eval()
    tokens = torch.tensor([[4, 5, 6, 3]], device="cuda")
    # Accelerate wraps the trained model in autocast. Match that precision
    # when comparing it with a freshly loaded adapter.
    with torch.no_grad(), torch.autocast("cuda", dtype=evaluation_dtype):
        expected = model(tokens).logits.float()
        with model.disable_adapter():
            expected_base = model(tokens).logits.float()
    adapter_dir = directory / "adapter"
    model.save_pretrained(
        adapter_dir, safe_serialization=True, save_embedding_layers=False
    )
    reloaded_base = LlamaForCausalLM(config).cuda()
    reloaded_base.load_state_dict(initial_base)
    reloaded = PeftModel.from_pretrained(
        reloaded_base, adapter_dir, local_files_only=True
    )
    for name, value in reloaded.named_parameters():
        torch.testing.assert_close(value, current[name], rtol=0, atol=0)
    reloaded.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=evaluation_dtype):
        torch.testing.assert_close(reloaded(tokens).logits.float(), expected)
        unloaded = reloaded.unload()
        torch.testing.assert_close(
            unloaded(tokens).logits.float(), expected_base
        )
    return {
        "status": "passed",
        "fixture": "tiny randomly initialized Llama; no pretrained assets",
        "training_steps": training_result.global_step,
        "training_loss": training_result.training_loss,
        "mixed_precision": str(evaluation_dtype),
        "trainable_parameters": sum(
            value.numel() for value in trainable.values()
        ),
        "base_weights_unchanged": True,
        "adapter_save_reload_disable_unload": "passed",
    }


def check_notebook() -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient

    code = (
        "import json, sys, torch\n"
        "assert torch.cuda.is_available()\n"
        "assert (torch.ones(2, device='cuda') + 1).sum().item() == 4\n"
        "print(json.dumps({'python': sys.executable, "
        "'gpu': torch.cuda.get_device_name(0)}))"
    )
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(code)]
    )
    client = NotebookClient(notebook, timeout=90, kernel_name="local-ai")
    client.execute(cwd=str(ROOT))
    output = "".join(
        item.get("text", "")
        for item in notebook.cells[0].outputs
        if item.output_type == "stream"
    )
    info = json.loads(output.strip())
    if Path(info["python"]).absolute() != Path(sys.executable).absolute():
        raise RuntimeError("Jupyter used a different Python interpreter")
    return {"status": "passed", "kernel": "local-ai", **info}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_environment()
    report: dict[str, Any] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "scope": "synthetic environment tests; no production model or dataset",
    }
    started = time.monotonic()
    work = ROOT / ".cache/phase2"
    work.mkdir(parents=True, exist_ok=True)
    try:
        with block_python_network() as attempts:
            report["checks"]["imports"] = check_imports()
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            for name, check in (
                ("nf4", check_four_bit),
                ("attention", check_attention),
                ("unsloth_kernel", check_unsloth_kernel),
            ):
                print(f"Checking {name}...", flush=True)
                report["checks"][name] = check()
            with tempfile.TemporaryDirectory(
                prefix="synthetic-", dir=work
            ) as temp:
                print(
                    "Checking synthetic TRL/LoRA step and adapter reload...",
                    flush=True,
                )
                report["checks"]["lora"] = check_lora(Path(temp))
            torch.cuda.synchronize()
            report["peak_torch_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / 1024**2, 2
            )
            report["python_network_attempts"] = attempts
            if attempts:
                raise RuntimeError(
                    "Framework tests attempted a network connection"
                )
        # Jupyter requires local sockets; restore socket functions first.
        print("Checking Jupyter kernel GPU execution...", flush=True)
        report["checks"]["jupyter"] = check_notebook()
        report["status"] = "passed"
    except Exception as exc:
        report.update(
            status="failed", error_type=type(exc).__name__, error=str(exc)
        )
        traceback.print_exc()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
