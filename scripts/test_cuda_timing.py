import time
import torch


def main():

    device = "cuda"

    x = torch.randn(
        4096,
        4096,
        device=device,
    )

    y = torch.randn(
        4096,
        4096,
        device=device,
    )

    # Warm-up
    for _ in range(3):
        _ = x @ y

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(
        enable_timing=True
    )

    end_event = torch.cuda.Event(
        enable_timing=True
    )

    wall_start = time.perf_counter()

    start_event.record()

    _ = x @ y

    end_event.record()

    torch.cuda.synchronize()

    wall_end = time.perf_counter()

    cuda_ms = start_event.elapsed_time(
        end_event
    )

    wall_ms = (
        wall_end - wall_start
    ) * 1000

    print(
        f"Wall time: {wall_ms:.3f} ms"
    )

    print(
        f"CUDA event time: {cuda_ms:.3f} ms"
    )


if __name__ == "__main__":
    main()