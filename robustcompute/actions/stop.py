from copy import deepcopy

from robustcompute.actions.base import InferenceAction
from robustcompute.core.task import Task
from robustcompute.core.context import InferenceContext
from robustcompute.core.result import ActionResult


class StopAction(InferenceAction):

    @property
    def name(self) -> str:
        return "STOP"

    def execute(
        self,
        task: Task,
        context: InferenceContext,
    ) -> ActionResult:

        result = deepcopy(
            context.base_result
        )

        result.action = self.name

        # STOP preserves the existing answer
        # but performs no ADDITIONAL inference.

        result.latency_ms = 0.0
        result.cuda_time_ms = 0.0

        result.input_tokens = 0
        result.output_tokens = 0
        result.decoding_steps = 0

        result.peak_memory_mb = 0.0
        result.peak_extra_memory_mb = 0.0

        result.avg_power_w = 0.0
        result.peak_power_w = 0.0
        result.energy_j = 0.0

        result.avg_gpu_utilization = 0.0
        result.peak_gpu_utilization = 0.0
        result.avg_memory_utilization = 0.0

        result.cost = 0.0

        result.metadata = {
            **result.metadata,
            "stopped_after_base": True,
        }

        return result