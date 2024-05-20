import math
from typing import Generic, Protocol, TypeVar

TState = TypeVar("TState")


class PEarlyStopper(Protocol[TState]):
    def step(self, score: float, state: TState | None = None) -> bool: ...

    def best_state(self) -> TState: ...

    def scores(self) -> list[float]: ...


class EarlyStopper(Generic[TState]):
    """
    Early stopper for training loops.

    Stops training if the score does not improve for a given number of steps.
    Improvement is defined as a decrease in score by at least `min_diff`.
    """

    def __init__(self, patients: int, min_diff: float):
        self.min_diff = min_diff
        self.patients = patients
        self.patients_counter = 0
        self.step_counter = 0
        self.min_score = math.inf
        self._best_state: TState | None = None

    def step(self, score: float, state: TState | None = None) -> bool:
        self.step_counter += 1
        if score + self.min_diff >= self.min_score:
            self.patients_counter += 1
        else:
            self.patients_counter = 0

        if score < self.min_score:
            self.min_score = score
            if state is not None:
                self._best_state = state

        if self.patients_counter > self.patients:
            return True
        else:
            return False

    def best_state(self) -> TState:
        if self._best_state is None:
            raise RuntimeError("State not stored.")
        return self._best_state
