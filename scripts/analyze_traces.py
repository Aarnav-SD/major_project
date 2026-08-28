import argparse

import pandas as pd


def analyze(path: str):

    df = pd.read_csv(path)

    n = len(df)

    print("\n========== TRACE SUMMARY ==========")
    print(f"Tasks: {n}")

    # ----------------------------------
    # QUALITY
    # ----------------------------------

    base_accuracy = df["base_quality"].mean()
    stop_accuracy = df["stop_quality"].mean()
    reason_accuracy = df["reason_quality"].mean()

    print("\n--- Accuracy ---")

    print(f"BASE:   {base_accuracy:.3f}")
    print(f"STOP:   {stop_accuracy:.3f}")
    print(f"REASON: {reason_accuracy:.3f}")

    # ----------------------------------
    # MARGINAL UTILITY
    # ----------------------------------

    positive = (df["delta_reason"] > 0).sum()
    zero = (df["delta_reason"] == 0).sum()
    negative = (df["delta_reason"] < 0).sum()

    print("\n--- REASON marginal quality ---")

    print(
        f"Positive: {positive}/{n} "
        f"({positive / n:.1%})"
    )

    print(
        f"Zero:     {zero}/{n} "
        f"({zero / n:.1%})"
    )

    print(
        f"Negative: {negative}/{n} "
        f"({negative / n:.1%})"
    )

    print(
        f"Mean ΔQ:  "
        f"{df['delta_reason'].mean():+.3f}"
    )

    # ----------------------------------
    # RESOURCE COST
    # ----------------------------------

    print("\n--- Average resource usage ---")

    print(
        f"BASE tokens:   "
        f"{df['base_tokens'].mean():.1f}"
    )

    print(
        f"REASON tokens: "
        f"{df['reason_tokens'].mean():.1f}"
    )

    print(
        f"BASE latency:   "
        f"{df['base_latency_ms'].mean():.1f} ms"
    )

    print(
        f"REASON latency: "
        f"{df['reason_latency_ms'].mean():.1f} ms"
    )

    # ----------------------------------
    # INTERESTING CASES
    # ----------------------------------

    print("\n--- REASON helped ---")

    helped = df[df["delta_reason"] > 0]

    if len(helped) == 0:
        print("None")
    else:
        print(
            helped[
                [
                    "task_id",
                    "base_quality",
                    "reason_quality",
                    "delta_reason",
                ]
            ].to_string(index=False)
        )

    print("\n--- REASON hurt ---")

    hurt = df[df["delta_reason"] < 0]

    if len(hurt) == 0:
        print("None")
    else:
        print(
            hurt[
                [
                    "task_id",
                    "base_quality",
                    "reason_quality",
                    "delta_reason",
                ]
            ].to_string(index=False)
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/traces/gsm8k_base_performance_100_task.csv",
    )

    args = parser.parse_args()

    analyze(args.input)


if __name__ == "__main__":
    main()