from abc import ABC, abstractmethod
from typing import Any, Dict


class Scheduler(ABC):
    """Base class for schedulers.

    This class is more general than the PyTorch learning rate schedulers and supports
    scheduling of any value, such as weight decay, momentum, etc. It is recommended to
    use a PyTorch learning rate scheduler for learning rate scheduling.

    Subclasses must implement the `get_value` method.
    """

    def __init__(self) -> None:
        self._step_count: int
        self._initial_step()

    @abstractmethod
    def get_value(self, step: int) -> float:
        """Return the current value of the scheduler.

        Args:
            step:
                The step to get the value for.
        """
        ...

    # Methods below should not be overloaded and are used for better compatibility with
    # PyTorch learning rate schedulers and optimizers.

    @property
    def last_step(self) -> int:
        """Returns the last step count."""
        return self._step_count - 1

    @property
    def current_step(self) -> int:
        """Returns the current step count.

        Important: The step count starts at 1 for compatibility with PyTorch learning
        rate schedulers.
        """
        return self._step_count

    def step(self) -> None:
        """Perform a step.

        Important: This method should not be overloaded by subclasses. Instead,
        subclasses must implement the `get_value` method which takes an optional `step`
        argument.
        """
        self._step_count += 1

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the scheduler as a dict.

        It contains an entry for every variable in self.__dict__.
        """
        return self.__dict__.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the scheduler from a dict.

        Args:
            state_dict:
                Must be an object returned from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def _initial_step(self) -> None:
        """Initialize step counts and perform a step."""
        # This logic is here to have the same behavior as the PyTorch learning rate
        # schedulers. See:
        # https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
        self._step_count = 0
        self.step()
