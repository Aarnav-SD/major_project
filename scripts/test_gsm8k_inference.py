from robustcompute.datasets.gsm8k import GSM8KDataset
from robustcompute.models.llm import LocalLLM
from robustcompute.evaluation.gsm8k import score_gsm8k


def main():

    dataset = GSM8KDataset(
        split="train"
    )

    task = dataset[0]

    llm = LocalLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256
    )

    prompt = f"""
Solve the following math problem.

Question:
{task.question}

Return your final answer in this exact format:
FINAL_ANSWER: <number>
"""

    result = llm.generate(
        prompt=prompt,
        task_id=task.task_id,
        action="BASE"
    )

    result.quality = score_gsm8k(
        result.answer,
        task.ground_truth
    )

    print("\nQUESTION")
    print(task.question)

    print("\nGROUND TRUTH")
    print(task.ground_truth)

    print("\nMODEL ANSWER")
    print(result.answer)

    print("\nQUALITY")
    print(result.quality)

    print("\nLATENCY")
    print(result.latency_ms, "ms")

    print("\nTOKENS")
    print(
        "Input:",
        result.input_tokens,
        "Output:",
        result.output_tokens
    )


if __name__ == "__main__":
    main()