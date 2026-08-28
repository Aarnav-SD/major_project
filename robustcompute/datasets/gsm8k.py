import re

from datasets import load_dataset

from robustcompute.core.task import Task


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    GSM8K solutions normally end with:

        #### final_answer

    Example:
        #### 72
    """

    match = re.search(
        r"####\s*(.+)",
        answer_text
    )

    if match is None:
        raise ValueError(
            f"Could not extract GSM8K answer from:\n{answer_text}"
        )

    answer = match.group(1).strip()

    # GSM8K sometimes uses commas in numbers such as:
    # 1,234
    answer = answer.replace(",", "")

    return answer


class GSM8KDataset:
    def __init__(self, split: str = "train"):
        self.split = split

        dataset = load_dataset(
            "openai/gsm8k",
            "main"
        )

        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Task:
        row = self.data[idx]

        full_solution = row["answer"]

        final_answer = extract_gsm8k_answer(
            full_solution
        )

        return Task(
            task_id=f"gsm8k_{self.split}_{idx}",
            dataset="gsm8k",
            question=row["question"],
            ground_truth=final_answer,
            metadata={
                "split": self.split,
                "index": idx,
                "reference_solution": full_solution
            }
        )