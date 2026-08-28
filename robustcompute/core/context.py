from dataclasses import dataclass

from robustcompute.core.result import ActionResult


@dataclass
class InferenceContext:
    base_result: ActionResult
    budget: float | None = None