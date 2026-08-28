from robustcompute.actions.base import InferenceAction
from robustcompute.core.task import Task
from robustcompute.core.context import InferenceContext
from robustcompute.core.result import ActionResult
from robustcompute.evaluation.gsm8k import score_gsm8k


class ReasonAction(InferenceAction):

    def __init__(
        self,
        llm,
        max_new_tokens: int = 512
    ):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        return "REASON"

    def execute(
        self,
        task: Task,
        context: InferenceContext
    ) -> ActionResult:

        prompt = f"""
Solve the following mathematical problem carefully.

Use additional reasoning if necessary. Work through the problem
step by step and verify your calculations before answering.

Question:
{task.question}

Return your final answer in exactly this format:

FINAL_ANSWER: <number>
"""

        result = self.llm.generate(
            prompt=prompt,
            task_id=task.task_id,
            action=self.name,
            max_new_tokens=self.max_new_tokens
        )

        result.quality = score_gsm8k(
            result.answer,
            task.ground_truth
        )

        return result