import re


def extract_numeric_answer(text: str) -> str | None:
    """
    Extract a model's final numeric answer.

    Preferred format:
        FINAL_ANSWER: 72

    Falls back to the last number in the response.
    """

    final_match = re.search(
        r"FINAL_ANSWER\s*:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )

    if final_match:
        return final_match.group(1).replace(",", "")

    numbers = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        text
    )

    if not numbers:
        return None

    return numbers[-1].replace(",", "")


def score_gsm8k(
    predicted_answer: str,
    ground_truth: str
) -> float:

    predicted = extract_numeric_answer(predicted_answer)

    if predicted is None:
        return 0.0

    return float(predicted == ground_truth)