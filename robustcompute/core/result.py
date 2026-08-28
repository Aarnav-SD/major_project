from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    task_id: str
    action: str
    answer: str
    quality: float

    # -----------------------------------
    # Layer 1: Model workload
    # -----------------------------------

    input_tokens: int = 0
    output_tokens: int = 0

    # For an autoregressive decoder, one output token
    # corresponds approximately to one decoding step.
    decoding_steps: int = 0

    # -----------------------------------
    # Layer 2: Runtime telemetry
    # -----------------------------------

    latency_ms: float = 0.0
    cuda_time_ms: float = 0.0
    inference_wall_ms: float = 0.0

    peak_memory_mb: float = 0.0
    peak_extra_memory_mb: float = 0.0

    # -----------------------------------
    # Layer 3: GPU telemetry
    # -----------------------------------

    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    energy_j: float = 0.0

    avg_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0

    avg_memory_utilization: float = 0.0

    # -----------------------------------
    # Cost / future extensions
    # -----------------------------------

    cost: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens