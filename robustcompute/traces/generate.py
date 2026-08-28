import csv
from pathlib import Path

from robustcompute.actions.base_inference import BaseInference
from robustcompute.core.context import InferenceContext


def generate_gsm8k_traces(
    dataset,
    base_inference: BaseInference,
    stop_action,
    reason_action,
    output_path: str,
    limit: int = 20,
):
    """
    Generate counterfactual GSM8K traces.

    For every task we observe all currently available
    inference paths:

        BASE -> STOP
        BASE -> REASON

    This is an offline counterfactual data-generation
    procedure. Later online-routing experiments will
    reveal only the outcome of the selected action.
    """

    rows = []

    total = min(
        limit,
        len(dataset),
    )

    for i in range(total):

        task = dataset[i]

        print(
            f"\n[{i + 1}/{total}] "
            f"{task.task_id}"
        )

        # =================================
        # BASE
        # =================================

        base_result = (
            base_inference.execute(task)
        )

        context = InferenceContext(
            base_result=base_result
        )

        # =================================
        # ACTIONS
        # =================================

        stop_result = stop_action.execute(
            task,
            context,
        )

        reason_result = reason_action.execute(
            task,
            context,
        )

        # =================================
        # MARGINAL QUALITY
        # =================================

        delta_stop = (
            stop_result.quality
            - base_result.quality
        )

        delta_reason = (
            reason_result.quality
            - base_result.quality
        )

        # =================================
        # END-TO-END PATH COSTS
        # =================================

        # STOP adds no inference work, but
        # the BASE computation is still a
        # sunk request cost.

        stop_path_total_tokens = (
            base_result.total_tokens
            + stop_result.total_tokens
        )

        stop_path_total_latency_ms = (
            base_result.latency_ms
            + stop_result.latency_ms
        )

        stop_path_total_energy_j = (
            base_result.energy_j
            + stop_result.energy_j
        )

        # REASON is executed after BASE from
        # the controller's cost perspective.

        reason_path_total_tokens = (
            base_result.total_tokens
            + reason_result.total_tokens
        )

        reason_path_total_latency_ms = (
            base_result.latency_ms
            + reason_result.latency_ms
        )

        reason_path_total_energy_j = (
            base_result.energy_j
            + reason_result.energy_j
        )

        # =================================
        # TRACE ROW
        # =================================

        row = {

            # ---------------------------------
            # Task
            # ---------------------------------

            "task_id": task.task_id,
            "dataset": task.dataset,
            "question": task.question,
            "ground_truth": task.ground_truth,

            # =================================
            # BASE — OUTCOME
            # =================================

            "base_answer":
                base_result.answer,

            "base_quality":
                base_result.quality,

            # ---------------------------------
            # BASE — Layer 1: workload
            # ---------------------------------

            "base_input_tokens":
                base_result.input_tokens,

            "base_output_tokens":
                base_result.output_tokens,

            "base_total_tokens":
                base_result.total_tokens,

            "base_decoding_steps":
                base_result.decoding_steps,

            # ---------------------------------
            # BASE — Layer 2: runtime
            # ---------------------------------

            "base_latency_ms":
                base_result.latency_ms,

            "base_cuda_time_ms":
                base_result.cuda_time_ms,

            "base_inference_wall_ms":
                base_result.inference_wall_ms,

            "base_peak_memory_mb":
                base_result.peak_memory_mb,

            "base_peak_extra_memory_mb":
                base_result.peak_extra_memory_mb,

            # ---------------------------------
            # BASE — Layer 3: GPU
            # ---------------------------------

            "base_avg_power_w":
                base_result.avg_power_w,

            "base_peak_power_w":
                base_result.peak_power_w,

            "base_energy_j":
                base_result.energy_j,

            "base_avg_gpu_utilization":
                base_result.avg_gpu_utilization,

            "base_peak_gpu_utilization":
                base_result.peak_gpu_utilization,

            "base_avg_memory_utilization":
                base_result.avg_memory_utilization,

            # =================================
            # STOP — OUTCOME
            # =================================

            "stop_quality":
                stop_result.quality,

            "delta_stop":
                delta_stop,

            # ---------------------------------
            # STOP — incremental action cost
            # ---------------------------------

            "stop_additional_tokens":
                stop_result.total_tokens,

            "stop_additional_latency_ms":
                stop_result.latency_ms,

            "stop_additional_cuda_time_ms":
                stop_result.cuda_time_ms,

            "stop_additional_energy_j":
                stop_result.energy_j,

            # ---------------------------------
            # BASE -> STOP total path
            # ---------------------------------

            "stop_path_total_tokens":
                stop_path_total_tokens,

            "stop_path_total_latency_ms":
                stop_path_total_latency_ms,

            "stop_path_total_energy_j":
                stop_path_total_energy_j,

            # =================================
            # REASON — OUTCOME
            # =================================

            "reason_answer":
                reason_result.answer,

            "reason_quality":
                reason_result.quality,

            "delta_reason":
                delta_reason,

            # ---------------------------------
            # REASON — Layer 1: workload
            # ---------------------------------

            "reason_input_tokens":
                reason_result.input_tokens,

            "reason_output_tokens":
                reason_result.output_tokens,

            "reason_total_tokens":
                reason_result.total_tokens,

            "reason_decoding_steps":
                reason_result.decoding_steps,

            # ---------------------------------
            # REASON — Layer 2: runtime
            # ---------------------------------

            "reason_latency_ms":
                reason_result.latency_ms,

            "reason_cuda_time_ms":
                reason_result.cuda_time_ms,

            "reason_inference_wall_ms":
                reason_result.inference_wall_ms,

            "reason_peak_memory_mb":
                reason_result.peak_memory_mb,

            "reason_peak_extra_memory_mb":
                reason_result.peak_extra_memory_mb,

            # ---------------------------------
            # REASON — Layer 3: GPU
            # ---------------------------------

            "reason_avg_power_w":
                reason_result.avg_power_w,

            "reason_peak_power_w":
                reason_result.peak_power_w,

            "reason_energy_j":
                reason_result.energy_j,

            "reason_avg_gpu_utilization":
                reason_result.avg_gpu_utilization,

            "reason_peak_gpu_utilization":
                reason_result.peak_gpu_utilization,

            "reason_avg_memory_utilization":
                reason_result.avg_memory_utilization,

            # ---------------------------------
            # BASE -> REASON total path
            # ---------------------------------

            "reason_path_total_tokens":
                reason_path_total_tokens,

            "reason_path_total_latency_ms":
                reason_path_total_latency_ms,

            "reason_path_total_energy_j":
                reason_path_total_energy_j,
        }

        rows.append(row)

        # =================================
        # LIVE DIAGNOSTICS
        # =================================

        print(
            f"BASE={base_result.quality:.0f} "
            f"("
            f"{base_result.output_tokens} out tok, "
            f"{base_result.latency_ms:.0f} ms, "
            f"{base_result.energy_j:.1f} J"
            f") | "
            f"STOP={stop_result.quality:.0f} | "
            f"REASON={reason_result.quality:.0f} "
            f"("
            f"{reason_result.output_tokens} out tok, "
            f"{reason_result.latency_ms:.0f} ms, "
            f"{reason_result.energy_j:.1f} J"
            f") | "
            f"ΔReason={delta_reason:+.0f}"
        )

    save_traces(
        rows,
        output_path,
    )

    return rows


def save_traces(
    rows,
    output_path: str,
):

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved {len(rows)} "
        f"traces to:"
    )

    print(path)