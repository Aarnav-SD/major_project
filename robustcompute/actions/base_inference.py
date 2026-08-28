from robustcompute.core.task import Task
from robustcompute.core.result import ActionResult
from robustcompute.evaluation.gsm8k import score_gsm8k


class BaseInference:
    """
    Performs the initial low-cost inference attempt.

    BASE is not a routing action. It establishes the initial
    answer/state from which STOP, REASON, RETRIEVE, etc. are
    evaluated.
    """

    def __init__(
        self,
        llm,
        max_new_tokens: int = 96,
    ):
        self.llm = llm
        self.max_new_tokens = max_new_tokens

    def execute(
        self,
        task: Task,
    ) -> ActionResult:

        if task.dataset != "gsm8k":
            raise ValueError(
                f"BaseInference does not yet support "
                f"dataset: {task.dataset}"
            )

        prompt = f"""
Solve the following math problem using only the minimum calculation needed.

Question:
{task.question}

Use short mathematical expressions only.
Do not explain your reasoning in prose.

End with exactly:
FINAL_ANSWER: <number>
""".strip()

        result = self.llm.generate(
            prompt=prompt,
            task_id=task.task_id,
            action="BASE",
            max_new_tokens=self.max_new_tokens,
        )

        result.quality = score_gsm8k(
            result.answer,
            task.ground_truth,
        )

        return result