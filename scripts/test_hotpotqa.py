from robustcompute.datasets.hotpotqa import (
    HotpotQADataset,
)


def main():

    print("Loading HotpotQA...")

    dataset = HotpotQADataset(
        split="train"
    )

    print(
        "Dataset size:",
        len(dataset)
    )

    task = dataset[0]

    print("\nTask ID:")
    print(task.task_id)

    print("\nQuestion:")
    print(task.question)

    print("\nGround truth:")
    print(task.ground_truth)

    print("\nContext:")
    print(task.metadata["context"])

    print("\nSupporting facts:")
    print(
        task.metadata[
            "supporting_facts"
        ]
    )


if __name__ == "__main__":
    main()