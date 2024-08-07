from lightly.schedulers.scheduler import Scheduler


class DummyScheduler(Scheduler):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def get_value(self, step: int) -> float:
        return step * self.value


class TestScheduler:
    def test_get_value(self) -> None:
        scheduler = DummyScheduler(value=1.0)
        assert scheduler.get_value(0) == 0.0
        assert scheduler.get_value(1) == 1.0
        assert scheduler.get_value(10) == 10.0

    def test_step(self) -> None:
        scheduler = DummyScheduler(value=1.0)
        assert scheduler.last_step == 0
        assert scheduler.current_step == 1
        scheduler.step()
        assert scheduler.last_step == 1
        assert scheduler.current_step == 2
        scheduler.step()
        assert scheduler.last_step == 2
        assert scheduler.current_step == 3

    def test_state_dict(self) -> None:
        scheduler = DummyScheduler(value=1.0)
        state_dict = scheduler.state_dict()
        assert state_dict == {"value": 1.0, "_step_count": 1}

    def test_load_state_dict(self) -> None:
        scheduler = DummyScheduler(value=1.0)
        state_dict = {"value": 2.0, "_step_count": 2}
        scheduler.load_state_dict(state_dict)
        assert scheduler.value == 2.0
        assert scheduler.current_step == 2
