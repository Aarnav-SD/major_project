from robustcompute.models.llm import LocalLLM


def main():

    llm = LocalLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=64
    )

    result = llm.generate(
        prompt="What is 17 + 25? Give only the final answer.",
        task_id="test_001",
        action="BASE"
    )

    print("\nAnswer:")
    print(result.answer)

    print("\nLatency:")
    print(result.latency_ms, "ms")

    print("\nInput tokens:")
    print(result.input_tokens)

    print("\nOutput tokens:")
    print(result.output_tokens)


if __name__ == "__main__":
    main()