from robustcompute.datasets.gsm8k import GSM8KDataset
from robustcompute.models.llm import LocalLLM

from robustcompute.actions.stop import StopAction
from robustcompute.actions.reason import ReasonAction

from robustcompute.traces.generate import (
    generate_gsm8k_traces
)


def main():

    dataset = GSM8KDataset(
        split="train"
    )

    llm = LocalLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256
    )

    stop_action = StopAction()

    reason_action = ReasonAction(
        llm=llm,
        max_new_tokens=512
    )

    generate_gsm8k_traces(
        dataset=dataset,
        llm=llm,
        stop_action=stop_action,
        reason_action=reason_action,
        output_path="data/traces/gsm8k_initial.csv",
        limit=10,
    )


if __name__ == "__main__":
    main()