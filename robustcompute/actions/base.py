from abc import ABC, abstractmethod

from robustcompute.core.task import Task
from robustcompute.core.context import InferenceContext
from robustcompute.core.result import ActionResult


class InferenceAction(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(
        self,
        task: Task,
        context: InferenceContext
    ) -> ActionResult:
        pass