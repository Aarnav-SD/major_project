from robustcompute.datasets.gsm8k import GSM8KDataset
from robustcompute.models.llm import LocalLLM

from robustcompute.actions.base_inference import BaseInference
from robustcompute.actions.stop import StopAction
from robustcompute.actions.reason import ReasonAction

from robustcompute.traces.generate import (
    generate_gsm8k_traces
)


def main():

    # =================================
    # DATASET
    # =================================

    dataset = GSM8KDataset(
        split="train"
    )

    # =================================
    # MODEL
    # =================================

    llm = LocalLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256,
    )

    # =================================
    # INITIAL INFERENCE
    # =================================

    base_inference = BaseInference(
        llm=llm,
        max_new_tokens=96,
    )

    # =================================
    # ACTIONS
    # =================================

    stop_action = StopAction()

    reason_action = ReasonAction(
        llm=llm,
        max_new_tokens=512,
    )

    # =================================
    # GENERATE
    # =================================

    generate_gsm8k_traces(
        dataset=dataset,
        base_inference=base_inference,
        stop_action=stop_action,
        reason_action=reason_action,

        output_path=(
            "data/traces/"
            "gsm8k_base_performance_100_task.csv"
        ),

        limit=100,
    )


if __name__ == "__main__":
    main()