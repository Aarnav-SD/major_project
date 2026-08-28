from robustcompute.datasets.gsm8k import GSM8KDataset


def main():
    print("Starting GSM8K test...")
    dataset = GSM8KDataset(split="train")

    print(f"Dataset size: {len(dataset)}")

    task = dataset[0]

    print("\nTask ID:")
    print(task.task_id)

    print("\nDataset:")
    print(task.dataset)

    print("\nQuestion:")
    print(task.question)

    print("\nGround truth:")
    print(task.ground_truth)

    print("\nMetadata:")
    print(task.metadata)


if __name__ == "__main__":
    main()