from robustcompute.datasets.gsm8k import (
    GSM8KDataset,
    extract_gsm8k_answer
)


def test_extract_gsm8k_answer():
    text = """
    Natalia sold 48/2 = 24 clips in May.
    She therefore sold 48+24 = 72 clips.
    #### 72
    """

    assert extract_gsm8k_answer(text) == "72"


def test_extract_answer_with_comma():
    text = """
    Some calculation.
    #### 1,234
    """

    assert extract_gsm8k_answer(text) == "1234"


def test_dataset_returns_task():
    dataset = GSM8KDataset(split="train")

    task = dataset[0]

    assert task.dataset == "gsm8k"
    assert task.question
    assert task.ground_truth
    assert "reference_solution" in task.metadata