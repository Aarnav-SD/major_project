import csv
from pathlib import Path

from robustcompute.core.context import InferenceContext
from robustcompute.evaluation.gsm8k import score_gsm8k


def generate_gsm8k_traces(
    dataset,
    llm,
    stop_action,
    reason_action,
    output_path: str,
    limit: int = 20,
):
    rows = []

    total = min(limit, len(dataset))

    for i in range(total):
        task = dataset[i]

        print(f"\n[{i + 1}/{total}] {task.task_id}")

        # ----------------------------
        # BASE
        # ----------------------------

        base_prompt = f"""
Solve the following math problem.

Question:
{task.question}

Return your final answer in this exact format:
FINAL_ANSWER: <number>
"""

        base_result = llm.generate(
            prompt=base_prompt,
            task_id=task.task_id,
            action="BASE",
            max_new_tokens=256,
        )

        base_result.quality = score_gsm8k(
            base_result.answer,
            task.ground_truth,
        )

        context = InferenceContext(
            base_result=base_result
        )

        # ----------------------------
        # ACTIONS
        # ----------------------------

        stop_result = stop_action.execute(
            task,
            context,
        )

        reason_result = reason_action.execute(
            task,
            context,
        )

        # ----------------------------
        # MARGINAL QUALITY
        # ----------------------------

        delta_stop = (
            stop_result.quality
            - base_result.quality
        )

        delta_reason = (
            reason_result.quality
            - base_result.quality
        )

        # ----------------------------
        # COST PROXIES
        # ----------------------------

        base_tokens = (
            base_result.input_tokens
            + base_result.output_tokens
        )

        reason_tokens = (
            reason_result.input_tokens
            + reason_result.output_tokens
        )

        # STOP performs no extra work.
        stop_tokens = 0

        row = {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "question": task.question,
            "ground_truth": task.ground_truth,

            "base_answer": base_result.answer,
            "base_quality": base_result.quality,
            "base_latency_ms": base_result.latency_ms,
            "base_tokens": base_tokens,

            "stop_quality": stop_result.quality,
            "stop_latency_ms": stop_result.latency_ms,
            "stop_tokens": stop_tokens,
            "delta_stop": delta_stop,

            "reason_answer": reason_result.answer,
            "reason_quality": reason_result.quality,
            "reason_latency_ms": reason_result.latency_ms,
            "reason_tokens": reason_tokens,
            "delta_reason": delta_reason,
        }

        rows.append(row)

        print(
            f"BASE={base_result.quality:.0f} | "
            f"STOP={stop_result.quality:.0f} | "
            f"REASON={reason_result.quality:.0f} | "
            f"ΔReason={delta_reason:+.0f}"
        )

    save_traces(rows, output_path)

    return rows


def save_traces(rows, output_path: str):
    path = Path(output_path)

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

    print(f"\nSaved {len(rows)} traces to:")
    print(path)