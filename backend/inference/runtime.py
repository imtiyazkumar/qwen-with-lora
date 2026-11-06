"""Load the selected text model and generate answers entirely on the GPU."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.services.environment import ROOT, configure_environment

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSettings:
    repo_id: str
    revision: str
    license: str
    local_path: str
    runtime: str
    dtype: str
    device: str
    attention_implementation: str
    max_context_tokens: int
    max_input_tokens: int
    max_new_tokens: int
    enable_thinking: bool

    @classmethod
    def read(cls, path: Path | None = None) -> ModelSettings:
        settings = cls(
            **json.loads((path or ROOT / "config/model.json").read_text())
        )
        if settings.dtype != "bfloat16" or settings.device != "cuda:0":
            raise ValueError(
                "This verified configuration requires BF16 on cuda:0"
            )
        if min(settings.max_input_tokens, settings.max_new_tokens) < 1:
            raise ValueError("Token limits must be positive")
        if (
            settings.max_input_tokens + settings.max_new_tokens
            > settings.max_context_tokens
        ):
            raise ValueError(
                "Input plus output limits exceed the context limit"
            )
        return settings

    @property
    def path(self) -> Path:
        return ROOT / self.local_path


class LocalModel:
    """One local model instance, using no CPU or disk offload."""

    def __init__(self, settings: ModelSettings | None = None) -> None:
        self.settings = settings or ModelSettings.read()
        configure_environment()
        # Initialize Unsloth before Transformers to enable bundled kernels.
        import torch
        import unsloth  # noqa: F401
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self.torch = torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU supporting BF16 is required")
        if not (self.settings.path / "config.json").is_file():
            raise FileNotFoundError(
                "Model assets missing; run scripts/download_model.py"
            )
        self.config = AutoConfig.from_pretrained(
            self.settings.path, local_files_only=True, trust_remote_code=False
        )
        if self.config.model_type != "qwen3_5":
            raise ValueError("Unexpected checkpoint architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.path,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        if not self.tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        LOGGER.info(
            "Loading local BF16 text model from %s", self.settings.path
        )
        self.model, self.loading_info = AutoModelForCausalLM.from_pretrained(
            self.settings.path,
            dtype=torch.bfloat16,
            device_map={"": self.settings.device},
            attn_implementation=self.settings.attention_implementation,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        self.model.eval()
        torch.cuda.synchronize()
        self.load_seconds = time.monotonic() - started
        if self.loading_info.get("missing_keys") or self.loading_info.get(
            "mismatched_keys"
        ):
            raise RuntimeError(
                f"Checkpoint loading failed: {self.loading_info}"
            )
        if {
            str(parameter.device) for parameter in self.model.parameters()
        } != {"cuda:0"}:
            raise RuntimeError("Unexpected CPU or disk offload")
        if self.model.config.model_type != "qwen3_5_text":
            raise RuntimeError("Expected the text-only Qwen3.5 model")

    def memory(self) -> dict[str, float]:
        torch = self.torch
        free, total = torch.cuda.mem_get_info()
        return {
            "torch_allocated_mib": round(
                torch.cuda.memory_allocated() / 1024**2, 1
            ),
            "torch_reserved_mib": round(
                torch.cuda.memory_reserved() / 1024**2, 1
            ),
            "peak_torch_allocated_mib": round(
                torch.cuda.max_memory_allocated() / 1024**2, 1
            ),
            "peak_torch_reserved_mib": round(
                torch.cuda.max_memory_reserved() / 1024**2, 1
            ),
            "gpu_free_mib": round(free / 1024**2, 1),
            "gpu_used_mib": round((total - free) / 1024**2, 1),
            "gpu_total_mib": round(total / 1024**2, 1),
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        torch = self.torch
        limit = (
            self.settings.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        if not 1 <= limit <= self.settings.max_new_tokens:
            raise ValueError("Requested output exceeds configured token limit")
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.settings.enable_thinking,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens > self.settings.max_input_tokens:
            raise ValueError(
                "Conversation exceeds the input budget; clear or shorten it"
            )
        if prompt_tokens + limit > self.settings.max_context_tokens:
            raise ValueError("Prompt plus response exceeds the context budget")
        inputs = inputs.to(self.settings.device)

        class TokenTimer:
            def __init__(self) -> None:
                self.prompt = True
                self.times: list[float] = []

            def put(self, value: Any) -> None:
                if self.prompt:
                    self.prompt = False
                else:
                    self.times.append(time.monotonic())

            def end(self) -> None:
                pass

        timer = TokenTimer()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=timer,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
        count = len(generated)
        duration = (
            timer.times[-1] - timer.times[0] if len(timer.times) > 1 else 0
        )
        return {
            "answer": text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second_including_prefill": round(count / elapsed, 2),
            "decode_tokens_per_second": round((count - 1) / duration, 2)
            if duration
            else None,
            "time_to_first_token_seconds": round(timer.times[0] - started, 3)
            if timer.times
            else None,
            "output_truncated": count == limit
            and int(generated[-1]) != self.tokenizer.eos_token_id,
            "memory": self.memory(),
            "knowledge_mode": "general_model_or_explicit_prompt_context",
        }
