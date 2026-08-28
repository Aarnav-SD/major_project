from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    task_id: str
    dataset: str
    question: str
    ground_truth: Any
    metadata: dict[str, Any] = field(default_factory=dict)