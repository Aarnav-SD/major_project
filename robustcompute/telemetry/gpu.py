import threading
import time
from dataclasses import dataclass
from statistics import mean

import pynvml


@dataclass
class GPUSample:
    timestamp: float

    power_w: float | None
    gpu_utilization: float | None
    memory_utilization: float | None


@dataclass
class GPUMetrics:
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    energy_j: float = 0.0

    avg_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0

    avg_memory_utilization: float = 0.0

    sample_count: int = 0


class GPUMonitor:
    """
    Periodically samples NVIDIA GPU telemetry using NVML.

    Intended to measure hardware activity during an inference action.
    """

    def __init__(
        self,
        device_index: int = 0,
        sampling_interval_s: float = 0.05,
    ):
        self.device_index = device_index
        self.sampling_interval_s = sampling_interval_s

        pynvml.nvmlInit()

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(
            device_index
        )

        self.samples: list[GPUSample] = []

        self._running = False
        self._thread = None

    def _get_power(self) -> float | None:

        try:
            # NVML returns milliwatts.
            power_mw = pynvml.nvmlDeviceGetPowerUsage(
                self.handle
            )

            return power_mw / 1000.0

        except pynvml.NVMLError:
            return None

    def _get_utilization(
        self
    ) -> tuple[float | None, float | None]:

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(
                self.handle
            )

            return (
                float(util.gpu),
                float(util.memory),
            )

        except pynvml.NVMLError:
            return None, None

    def _sample_loop(self):

        while self._running:

            timestamp = time.perf_counter()

            power = self._get_power()

            gpu_util, memory_util = (
                self._get_utilization()
            )

            self.samples.append(
                GPUSample(
                    timestamp=timestamp,
                    power_w=power,
                    gpu_utilization=gpu_util,
                    memory_utilization=memory_util,
                )
            )

            time.sleep(
                self.sampling_interval_s
            )

    def start(self):

        self.samples = []

        self._running = True

        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> GPUMetrics:

        self._running = False

        if self._thread is not None:
            self._thread.join()

        return self._calculate_metrics()

    def _calculate_metrics(
        self
    ) -> GPUMetrics:

        if not self.samples:
            return GPUMetrics()

        # ---------------------------------
        # Power
        # ---------------------------------

        power_samples = [
            sample.power_w
            for sample in self.samples
            if sample.power_w is not None
        ]

        if power_samples:
            avg_power = mean(power_samples)
            peak_power = max(power_samples)
        else:
            avg_power = 0.0
            peak_power = 0.0

        # ---------------------------------
        # GPU utilization
        # ---------------------------------

        gpu_util_samples = [
            sample.gpu_utilization
            for sample in self.samples
            if sample.gpu_utilization is not None
        ]

        if gpu_util_samples:
            avg_gpu_util = mean(
                gpu_util_samples
            )
            peak_gpu_util = max(
                gpu_util_samples
            )
        else:
            avg_gpu_util = 0.0
            peak_gpu_util = 0.0

        # ---------------------------------
        # Memory utilization
        # ---------------------------------

        memory_util_samples = [
            sample.memory_utilization
            for sample in self.samples
            if sample.memory_utilization is not None
        ]

        if memory_util_samples:
            avg_memory_util = mean(
                memory_util_samples
            )
        else:
            avg_memory_util = 0.0

        # ---------------------------------
        # Energy integration
        #
        # E = integral P(t) dt
        #
        # trapezoidal integration
        # ---------------------------------

        energy_j = 0.0

        valid_power_samples = [
            sample
            for sample in self.samples
            if sample.power_w is not None
        ]

        for previous, current in zip(
            valid_power_samples,
            valid_power_samples[1:],
        ):

            dt = (
                current.timestamp
                - previous.timestamp
            )

            avg_interval_power = (
                previous.power_w
                + current.power_w
            ) / 2.0

            energy_j += (
                avg_interval_power * dt
            )

        return GPUMetrics(
            avg_power_w=avg_power,
            peak_power_w=peak_power,
            energy_j=energy_j,

            avg_gpu_utilization=avg_gpu_util,
            peak_gpu_utilization=peak_gpu_util,

            avg_memory_utilization=(
                avg_memory_util
            ),

            sample_count=len(
                self.samples
            ),
        )

    def shutdown(self):

        try:
            pynvml.nvmlShutdown()

        except pynvml.NVMLError:
            pass