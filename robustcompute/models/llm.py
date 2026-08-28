import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from robustcompute.core.result import ActionResult
from robustcompute.telemetry.gpu import GPUMonitor


class LocalLLM:

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        device: str | None = None,
    ):

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        print(
            f"Loading model: {self.model_name}"
        )

        print(
            f"Using device: {self.device}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.gpu_monitor = None

        if self.device == "cuda":
            self.gpu_monitor = GPUMonitor(
                device_index=0,
                sampling_interval_s=0.05,
            )

    def generate(
        self,
        prompt: str,
        task_id: str,
        action: str,
        max_new_tokens: int | None = None,
    ) -> ActionResult:

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # =================================
        # TOKENIZATION
        # =================================

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[-1]

        # =================================
        # TELEMETRY DEFAULTS
        # =================================

        cuda_time_ms = 0.0
        inference_wall_ms = 0.0

        peak_memory_mb = 0.0
        peak_extra_memory_mb = 0.0

        gpu_metrics = None

        memory_before = 0

        # =================================
        # PREPARE CUDA TELEMETRY
        # =================================

        if self.device == "cuda":

            torch.cuda.synchronize()

            memory_before = torch.cuda.memory_allocated()

            torch.cuda.reset_peak_memory_stats()

            cuda_start = torch.cuda.Event(
                enable_timing=True
            )

            cuda_end = torch.cuda.Event(
                enable_timing=True
            )

        # =================================
        # SINGLE INFERENCE + TELEMETRY WINDOW
        # =================================

        try:

            if self.gpu_monitor:
                self.gpu_monitor.start()

            # Wall clock encloses the CUDA
            # inference interval.
            wall_start = time.perf_counter()

            if self.device == "cuda":
                cuda_start.record()

            with torch.inference_mode():

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            if self.device == "cuda":

                cuda_end.record()

                # Wait until all queued GPU work
                # has completed before stopping
                # the wall timer.
                torch.cuda.synchronize()

            wall_end = time.perf_counter()

        finally:

            if self.gpu_monitor:
                gpu_metrics = self.gpu_monitor.stop()

        # =================================
        # RUNTIME TELEMETRY
        # =================================

        latency_ms = (
            wall_end - wall_start
        ) * 1000.0

        inference_wall_ms = latency_ms

        if self.device == "cuda":

            cuda_time_ms = cuda_start.elapsed_time(
                cuda_end
            )

            peak_memory = (
                torch.cuda.max_memory_allocated()
            )

            peak_memory_mb = (
                peak_memory / (1024 ** 2)
            )

            peak_extra_memory_mb = max(
                0.0,
                (
                    peak_memory
                    - memory_before
                )
                / (1024 ** 2),
            )

        # =================================
        # DECODE RESULT
        # =================================

        generated_tokens = (
            outputs[0][input_tokens:]
        )

        output_tokens = len(
            generated_tokens
        )

        total_sequence_tokens = (
            input_tokens
            + output_tokens
        )

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        # =================================
        # RESULT
        # =================================

        return ActionResult(
            task_id=task_id,
            action=action,

            answer=answer,
            quality=0.0,

            # Layer 1: model workload
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            decoding_steps=output_tokens,

            # Layer 2: runtime
            latency_ms=latency_ms,
            cuda_time_ms=cuda_time_ms,
            inference_wall_ms=inference_wall_ms,

            peak_memory_mb=peak_memory_mb,
            peak_extra_memory_mb=(
                peak_extra_memory_mb
            ),

            # Layer 3: GPU hardware
            avg_power_w=(
                gpu_metrics.avg_power_w
                if gpu_metrics
                else 0.0
            ),

            peak_power_w=(
                gpu_metrics.peak_power_w
                if gpu_metrics
                else 0.0
            ),

            energy_j=(
                gpu_metrics.energy_j
                if gpu_metrics
                else 0.0
            ),

            avg_gpu_utilization=(
                gpu_metrics.avg_gpu_utilization
                if gpu_metrics
                else 0.0
            ),

            peak_gpu_utilization=(
                gpu_metrics.peak_gpu_utilization
                if gpu_metrics
                else 0.0
            ),

            avg_memory_utilization=(
                gpu_metrics.avg_memory_utilization
                if gpu_metrics
                else 0.0
            ),

            cost=0.0,

            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "max_new_tokens": max_new_tokens,
                "total_sequence_tokens":
                    total_sequence_tokens,

                "gpu_sample_count": (
                    gpu_metrics.sample_count
                    if gpu_metrics
                    else 0
                ),
            },
        )