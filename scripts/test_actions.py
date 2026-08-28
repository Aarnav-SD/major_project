from robustcompute.datasets.gsm8k import GSM8KDataset
from robustcompute.models.llm import LocalLLM
from robustcompute.evaluation.gsm8k import score_gsm8k

from robustcompute.core.context import InferenceContext

from robustcompute.actions.base_inference import BaseInference
from robustcompute.actions.stop import StopAction
from robustcompute.actions.reason import ReasonAction


def main():

    dataset = GSM8KDataset(split="train")
    task = dataset[0]

    llm = LocalLLM(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256
    )

    # ---------------------------------------
    # BASE INFERENCE
    # ---------------------------------------

    base_inference = BaseInference(
        llm=llm,
        max_new_tokens=64
    )

    base_result = base_inference.execute(task)

    # ---------------------------------------
    # CREATE CONTEXT
    # ---------------------------------------

    context = InferenceContext(
        base_result=base_result
    )

    # ---------------------------------------
    # ACTIONS
    # ---------------------------------------

    stop_action = StopAction()

    reason_action = ReasonAction(
        llm=llm,
        max_new_tokens=512
    )

    stop_result = stop_action.execute(
        task,
        context
    )

    reason_result = reason_action.execute(
        task,
        context
    )

    # ---------------------------------------
    # MARGINAL QUALITY
    # ---------------------------------------

    delta_stop = (
        stop_result.quality
        - base_result.quality
    )

    delta_reason = (
        reason_result.quality
        - base_result.quality
    )

    # ---------------------------------------
    # OUTPUT
    # ---------------------------------------

    print("\nQUESTION")
    print(task.question)

    print("\nGROUND TRUTH")
    print(task.ground_truth)

    print("\n================ BASE ================")
    print(base_result.answer)

    print("\n--- Outcome ---")
    print(
        "Quality:",
        base_result.quality
    )

    print("\n--- Model workload ---")
    print(
        "Input tokens:",
        base_result.input_tokens
    )
    print(
        "Output tokens:",
        base_result.output_tokens
    )
    print(
        "Decoding steps:",
        base_result.decoding_steps
    )

    print("\n--- Runtime ---")
    print(
        "Wall latency:",
        base_result.latency_ms,
        "ms"
    )
    print(
        "CUDA time:",
        base_result.cuda_time_ms,
        "ms"
    )

    print(
        "Peak VRAM:",
        base_result.peak_memory_mb,
        "MB"
    )

    print(
        "Peak extra VRAM:",
        base_result.peak_extra_memory_mb,
        "MB"
    )

    print("\n--- GPU hardware ---")

    print(
        "Average power:",
        base_result.avg_power_w,
        "W"
    )

    print(
        "Peak power:",
        base_result.peak_power_w,
        "W"
    )

    print(
        "Energy:",
        base_result.energy_j,
        "J"
    )

    print(
        "Average GPU utilization:",
        base_result.avg_gpu_utilization,
        "%"
    )

    print(
        "Peak GPU utilization:",
        base_result.peak_gpu_utilization,
        "%"
    )

    print(
        "Average memory utilization:",
        base_result.avg_memory_utilization,
        "%"
    )

    print("\n================ STOP ================")

    print("Quality:", stop_result.quality)
    print("Delta Q:", delta_stop)

    print(
        "Additional tokens:",
        stop_result.total_tokens
    )

    print(
        "Additional CUDA time:",
        stop_result.cuda_time_ms,
        "ms"
    )

    print(
        "Additional energy:",
        stop_result.energy_j,
        "J"
    )

    print("\n=============== REASON ===============")
    print(reason_result.answer)

    print("\nQuality:", reason_result.quality)
    print("Latency:", reason_result.latency_ms)

    print(
        "Tokens:",
        reason_result.input_tokens
        + reason_result.output_tokens
    )

    print("Delta Q:", delta_reason)

    print("\n--- Model workload ---")
    print(
        "Input tokens:",
        reason_result.input_tokens
    )
    print(
        "Output tokens:",
        reason_result.output_tokens
    )
    print(
        "Decoding steps:",
        reason_result.decoding_steps
    )

    print("\n--- Runtime ---")
    print(
        "Wall latency:",
        reason_result.latency_ms,
        "ms"
    )
    print(
        "CUDA time:",
        reason_result.cuda_time_ms,
        "ms"
    )

    print(
        "Peak VRAM:",
        reason_result.peak_memory_mb,
        "MB"
    )

    print(
        "Peak extra VRAM:",
        reason_result.peak_extra_memory_mb,
        "MB"
    )

    print("\n--- GPU hardware ---")

    print(
        "Average power:",
        reason_result.avg_power_w,
        "W"
    )

    print(
        "Peak power:",
        reason_result.peak_power_w,
        "W"
    )

    print(
        "Energy:",
        reason_result.energy_j,
        "J"
    )

    print(
        "Average GPU utilization:",
        reason_result.avg_gpu_utilization,
        "%"
    )

    print(
        "Peak GPU utilization:",
        reason_result.peak_gpu_utilization,
        "%"
    )

    print(
        "Average memory utilization:",
        reason_result.avg_memory_utilization,
        "%"
    )


if __name__ == "__main__":
    main()