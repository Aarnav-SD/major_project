from datasets import load_dataset

from robustcompute.core.task import Task


class HotpotQADataset:

    def __init__(
        self,
        split: str = "train",
    ):
        self.split = split

        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            trust_remote_code=True,
        )

        self.data = dataset[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Task:

        row = self.data[idx]

        return Task(
            task_id=f"hotpotqa_{self.split}_{idx}",
            dataset="hotpotqa",
            question=row["question"],
            ground_truth=row["answer"],
            metadata={
                "split": self.split,
                "index": idx,
                "context": row["context"],
                "supporting_facts": row[
                    "supporting_facts"
                ],
            },
        )