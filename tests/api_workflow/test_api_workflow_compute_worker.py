import json
import random
from typing import Any, List
from unittest import mock

import pydantic
import pytest
import pytest_mock

from lightly import api as lightly_api
from lightly.api import api_workflow_compute_worker
from lightly.openapi_generated.swagger_client import api as swagger_api
from lightly.openapi_generated.swagger_client import api_client as swagger_api_client
from lightly.openapi_generated.swagger_client import exceptions, models
from tests.api_workflow import mocked_api_workflow_client, utils


class TestApiWorkflowComputeWorker(mocked_api_workflow_client.MockedApiWorkflowSetup):
    def test_register_compute_worker(self):
        # default name
        worker_id = self.api_workflow_client.register_compute_worker()
        assert worker_id
        # custom name
        worker_id = self.api_workflow_client.register_compute_worker(name="my-worker")
        assert worker_id

    def test_delete_compute_worker(self):
        with mock.patch(
            "tests.api_workflow.mocked_api_workflow_client.MockedComputeWorkerApi"
            ".delete_docker_worker_registry_entry_by_id",
        ) as mock_delete_worker:
            self.api_workflow_client.delete_compute_worker("worker_id")
            mock_delete_worker.assert_called_once_with("worker_id")

    def test_create_compute_worker_config(self):
        config_id = self.api_workflow_client.create_compute_worker_config(
            worker_config={
                "training": {"task_name": "lightly_pretagging"},
            },
            lightly_config={
                "loader": {
                    "batch_size": 64,
                },
            },
            selection_config={
                "n_samples": 20,
                "strategies": [
                    {
                        "input": {
                            "type": "EMBEDDINGS",
                            "dataset_id": utils.generate_id(),
                            "tag_name": "some-tag-name",
                        },
                        "strategy": {"type": "SIMILARITY"},
                    },
                ],
            },
        )
        assert config_id

    def test_create_compute_worker_config__selection_config_is_class(self) -> None:
        config_id = self.api_workflow_client.create_compute_worker_config(
            worker_config={
                "pretagging": True,
            },
            lightly_config={
                "loader": {
                    "batch_size": 64,
                },
            },
            selection_config=models.SelectionConfig(
                n_samples=20,
                strategies=[
                    models.SelectionConfigEntry(
                        input=models.SelectionConfigEntryInput(
                            type=models.SelectionInputType.EMBEDDINGS,
                            dataset_id=utils.generate_id(),
                            tag_name="some-tag-name",
                        ),
                        strategy=models.SelectionConfigEntryStrategy(
                            type=models.SelectionStrategyType.SIMILARITY,
                        ),
                    )
                ],
            ),
        )
        assert config_id

    def test_create_compute_worker_config__all_none(self) -> None:
        config_id = self.api_workflow_client.create_compute_worker_config(
            worker_config=None,
            lightly_config=None,
            selection_config=None,
        )
        assert config_id

    def test_schedule_compute_worker_run(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={
                "pretagging": True,
            },
            lightly_config={
                "loader": {
                    "batch_size": 64,
                },
            },
        )
        assert scheduled_run_id

    def test_schedule_compute_worker_run__priority(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={},
            lightly_config={},
            priority=models.DockerRunScheduledPriority.HIGH,
        )
        assert scheduled_run_id

    def test_schedule_compute_worker_run__runs_on(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={}, lightly_config={}, runs_on=["AAA", "BBB"]
        )
        assert scheduled_run_id

    def test_get_compute_worker_ids(self):
        ids = self.api_workflow_client.get_compute_worker_ids()
        assert all(isinstance(id_, str) for id_ in ids)

    def test_get_compute_workers(self):
        workers = self.api_workflow_client.get_compute_workers()
        assert len(workers) == 1
        assert workers[0].name == "worker-name-1"
        assert workers[0].state == models.DockerWorkerState.OFFLINE
        assert workers[0].labels == ["label-1"]

    def test_get_compute_worker_runs(self):
        runs = self.api_workflow_client.get_compute_worker_runs()
        assert len(runs) > 0
        assert all(isinstance(run, models.DockerRunData) for run in runs)

    def test_get_scheduled_compute_worker_runs(self):
        with mock.patch(
            "tests.api_workflow.mocked_api_workflow_client.MockedComputeWorkerApi"
            ".get_docker_runs_scheduled_by_dataset_id",
        ) as mock_get_runs:
            self.api_workflow_client.get_scheduled_compute_worker_runs()
            mock_get_runs.assert_called_once_with(
                dataset_id=self.api_workflow_client.dataset_id
            )

        with mock.patch(
            "tests.api_workflow.mocked_api_workflow_client.MockedComputeWorkerApi"
            ".get_docker_runs_scheduled_by_dataset_id",
        ) as mock_get_runs:
            self.api_workflow_client.get_scheduled_compute_worker_runs(state="state")
            mock_get_runs.assert_called_once_with(
                dataset_id=self.api_workflow_client.dataset_id, state="state"
            )

    def _check_if_openapi_generated_obj_is_valid(self, obj) -> Any:
        api_client = swagger_api_client.ApiClient()

        obj_as_json = json.dumps(api_client.sanitize_for_serialization(obj))

        mocked_response = mock.MagicMock()
        mocked_response.data = obj_as_json
        obj_api = api_client.deserialize(mocked_response, type(obj).__name__)

        self.assertDictEqual(obj.to_dict(), obj_api.to_dict())

        return obj_api

    def test_selection_config(self):
        selection_config = models.SelectionConfig(
            n_samples=1,
            strategies=[
                models.SelectionConfigEntry(
                    input=models.SelectionConfigEntryInput(
                        type=models.SelectionInputType.EMBEDDINGS
                    ),
                    strategy=models.SelectionConfigEntryStrategy(
                        type=models.SelectionStrategyType.DIVERSITY,
                        stopping_condition_minimum_distance=-1,
                    ),
                ),
                models.SelectionConfigEntry(
                    input=models.SelectionConfigEntryInput(
                        type=models.SelectionInputType.SCORES,
                        task="my-classification-task",
                        score="uncertainty_margin",
                    ),
                    strategy=models.SelectionConfigEntryStrategy(
                        type=models.SelectionStrategyType.WEIGHTS
                    ),
                ),
                models.SelectionConfigEntry(
                    input=models.SelectionConfigEntryInput(
                        type=models.SelectionInputType.METADATA, key="lightly.sharpness"
                    ),
                    strategy=models.SelectionConfigEntryStrategy(
                        type=models.SelectionStrategyType.THRESHOLD,
                        threshold=20,
                        operation=models.SelectionStrategyThresholdOperation.BIGGER_EQUAL,
                    ),
                ),
                models.SelectionConfigEntry(
                    input=models.SelectionConfigEntryInput(
                        type=models.SelectionInputType.PREDICTIONS,
                        task="my_object_detection_task",
                        name=models.SelectionInputPredictionsName.CLASS_DISTRIBUTION,
                    ),
                    strategy=models.SelectionConfigEntryStrategy(
                        type=models.SelectionStrategyType.BALANCE,
                        target={"Ambulance": 0.2, "Bus": 0.4},
                    ),
                ),
            ],
        )
        config = models.DockerWorkerConfigV3(
            worker_type=models.DockerWorkerType.FULL, selection=selection_config
        )

        self._check_if_openapi_generated_obj_is_valid(config)


def test_selection_config_from_dict() -> None:
    dataset_id = utils.generate_id()
    cfg = {
        "n_samples": 10,
        "proportion_samples": 0.1,
        "strategies": [
            {
                "input": {
                    "type": "EMBEDDINGS",
                    "dataset_id": dataset_id,
                    "tag_name": "some-tag-name",
                },
                "strategy": {"type": "SIMILARITY"},
            },
            {
                "input": {
                    "type": "METADATA",
                    "key": "lightly.sharpness",
                },
                "strategy": {
                    "type": "THRESHOLD",
                    "threshold": 20,
                    "operation": "BIGGER",
                },
            },
        ],
    }
    selection_cfg = api_workflow_compute_worker.selection_config_from_dict(cfg)
    assert selection_cfg.n_samples == 10
    assert selection_cfg.proportion_samples == 0.1
    assert selection_cfg.strategies is not None
    assert len(selection_cfg.strategies) == 2
    assert selection_cfg.strategies[0].input.type == "EMBEDDINGS"
    assert selection_cfg.strategies[0].input.dataset_id == dataset_id
    assert selection_cfg.strategies[0].input.tag_name == "some-tag-name"
    assert selection_cfg.strategies[0].strategy.type == "SIMILARITY"
    assert selection_cfg.strategies[1].input.type == "METADATA"
    assert selection_cfg.strategies[1].input.key == "lightly.sharpness"
    assert selection_cfg.strategies[1].strategy.type == "THRESHOLD"
    assert selection_cfg.strategies[1].strategy.threshold == 20
    assert selection_cfg.strategies[1].strategy.operation == "BIGGER"
    # verify that original dict was not mutated
    assert isinstance(cfg["strategies"][0]["input"], dict)


def test_selection_config_from_dict__missing_strategies() -> None:
    cfg = {}
    with pytest.raises(
        pydantic.ValidationError,
        match=r"strategies\n  ensure this value has at least 1 items",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_key() -> None:
    cfg = {"strategies": [], "invalid-key": 0}
    with pytest.raises(
        pydantic.ValidationError,
        match=r"invalid-key\n  extra fields not permitted",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_stratey_key() -> None:
    cfg = {
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS"},
                "strategy": {"type": "DIVERSITY"},
                "invalid-key": {"type": ""},
            },
        ],
    }
    with pytest.raises(
        pydantic.ValidationError,
        match=r"invalid-key\n  extra fields not permitted",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_strategy_strategy_key() -> None:
    cfg = {
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS"},
                "strategy": {
                    "type": "DIVERSITY",
                    "stoppingConditionMinimumDistance": 0,
                },
            },
        ],
    }
    with pytest.raises(
        pydantic.ValidationError,
        match=r"stoppingConditionMinimumDistance\n  extra fields not permitted",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__multiple_references() -> None:
    """Test that conversion is successful if the dictionary contains multiple references
    to the same object.
    """
    strategy = {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}}
    cfg = {"strategies": [strategy, strategy]}
    selection_cfg = api_workflow_compute_worker.selection_config_from_dict(cfg)
    assert len(selection_cfg.strategies) == 2
    assert selection_cfg.strategies[0] == selection_cfg.strategies[1]


def test_get_scheduled_run_by_id() -> None:
    run_ids = [utils.generate_id() for _ in range(3)]
    scheduled_runs = [
        models.DockerRunScheduledData(
            id=run_id,
            dataset_id=utils.generate_id(),
            config_id=utils.generate_id(),
            priority=models.DockerRunScheduledPriority.MID,
            state=models.DockerRunScheduledState.OPEN,
            created_at=0,
            last_modified_at=1,
            runs_on=[],
        )
        for run_id in run_ids
    ]
    mocked_compute_worker_api = mock.MagicMock(
        get_docker_runs_scheduled_by_dataset_id=lambda dataset_id: scheduled_runs
    )
    mocked_api_client = mock.MagicMock(
        dataset_id="asdf", _compute_worker_api=mocked_compute_worker_api
    )

    scheduled_run_id = run_ids[2]
    scheduled_run_data = lightly_api.ApiWorkflowClient._get_scheduled_run_by_id(
        self=mocked_api_client, scheduled_run_id=scheduled_run_id
    )
    assert scheduled_run_data.id == scheduled_run_id


def test_get_scheduled_run_by_id_not_found() -> None:
    scheduled_runs = [
        models.DockerRunScheduledData(
            id=utils.generate_id(),
            dataset_id=utils.generate_id(),
            config_id=utils.generate_id(),
            priority=models.DockerRunScheduledPriority.LOW,
            state=models.DockerRunScheduledState.OPEN,
            created_at=0,
            last_modified_at=1,
            runs_on=[],
        )
        for _ in range(3)
    ]
    mocked_compute_worker_api = mock.MagicMock(
        get_docker_runs_scheduled_by_dataset_id=lambda dataset_id: scheduled_runs
    )
    mocked_api_client = mock.MagicMock(
        dataset_id="asdf", _compute_worker_api=mocked_compute_worker_api
    )

    scheduled_run_id = "id_5"
    with pytest.raises(
        exceptions.ApiException,
        match=f"No scheduled run found for run with scheduled_run_id='{scheduled_run_id}'.",
    ):
        lightly_api.ApiWorkflowClient._get_scheduled_run_by_id(
            self=mocked_api_client, scheduled_run_id=scheduled_run_id
        )


def test_get_compute_worker_state_and_message_OPEN() -> None:
    dataset_id = utils.generate_id()
    scheduled_run = models.DockerRunScheduledData(
        id=utils.generate_id(),
        dataset_id=dataset_id,
        config_id=utils.generate_id(),
        priority=models.DockerRunScheduledPriority.MID,
        state=models.DockerRunScheduledState.OPEN,
        created_at=0,
        last_modified_at=1,
        runs_on=["worker-label"],
    )

    def mocked_raise_exception(*args, **kwargs):
        raise exceptions.ApiException

    mocked_api_client = mock.MagicMock(
        dataset_id=dataset_id,
        _compute_worker_api=mock.MagicMock(
            get_docker_run_by_scheduled_id=mocked_raise_exception
        ),
        _get_scheduled_run_by_id=lambda id: scheduled_run,
    )

    run_info = lightly_api.ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == models.DockerRunScheduledState.OPEN
    assert run_info.message.startswith("Waiting for pickup by Lightly Worker.")
    assert run_info.in_end_state() == False


def test_create_docker_worker_config_v3_api_error() -> None:
    class HttpThing:
        def __init__(self, status, reason, data):
            self.status = status
            self.reason = reason
            self.data = data

        def getheaders(self):
            return []

    def mocked_raise_exception(*args, **kwargs):
        raise exceptions.ApiException(
            http_resp=HttpThing(
                403,
                "Not everything has a reason",
                '{"code": "ACCOUNT_SUBSCRIPTION_INSUFFICIENT", "error": "Your current plan allows for 1000000 samples but you tried to use 2000000 samples, please contact sales at sales@lightly.ai to upgrade your account."}',
            )
        )

    client = lightly_api.ApiWorkflowClient(token="123")
    client._dataset_id = utils.generate_id()
    client._compute_worker_api.create_docker_worker_config_v3 = mocked_raise_exception
    with pytest.raises(
        ValueError,
        match=r'Trying to schedule your job resulted in\n>> ACCOUNT_SUBSCRIPTION_INSUFFICIENT\n>> "Your current plan allows for 1000000 samples but you tried to use 2000000 samples, please contact sales at sales@lightly.ai to upgrade your account."\n>> Please fix the issue mentioned above and see our docs https://docs.lightly.ai/docs/all-configuration-options for more help.',
    ):
        r = client.create_compute_worker_config(
            selection_config={
                "n_samples": 2000000,
                "strategies": [
                    {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}}
                ],
            },
        )


def test_create_docker_worker_config_v3_5xx_api_error() -> None:
    class HttpThing:
        def __init__(self, status, reason, data):
            self.status = status
            self.reason = reason
            self.data = data

        def getheaders(self):
            return []

    def mocked_raise_exception(*args, **kwargs):
        raise exceptions.ApiException(
            http_resp=HttpThing(
                502,
                "Not everything has a reason",
                '{"code": "SOMETHING_BAD", "error": "Server pains"}',
            )
        )

    client = lightly_api.ApiWorkflowClient(token="123")
    client._dataset_id = utils.generate_id()
    client._compute_worker_api.create_docker_worker_config_v3 = mocked_raise_exception
    with pytest.raises(
        exceptions.ApiException,
        match=r"Server pains",
    ):
        r = client.create_compute_worker_config(
            selection_config={
                "n_samples": 20,
                "strategies": [
                    {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}}
                ],
            },
        )


def test_create_docker_worker_config_v3_no_body_api_error() -> None:
    def mocked_raise_exception(*args, **kwargs):
        raise exceptions.ApiException

    client = lightly_api.ApiWorkflowClient(token="123")
    client._dataset_id = utils.generate_id()
    client._compute_worker_api.create_docker_worker_config_v3 = mocked_raise_exception
    with pytest.raises(
        exceptions.ApiException,
    ):
        r = client.create_compute_worker_config(
            selection_config={
                "n_samples": 20,
                "strategies": [
                    {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}}
                ],
            },
        )


def test_get_compute_worker_state_and_message_CANCELED() -> None:
    def mocked_raise_exception(*args, **kwargs):
        raise exceptions.ApiException

    mocked_api_client = mock.MagicMock(
        dataset_id=utils.generate_id(),
        _compute_worker_api=mock.MagicMock(
            get_docker_run_by_scheduled_id=mocked_raise_exception
        ),
        _get_scheduled_run_by_id=mocked_raise_exception,
    )
    run_info = lightly_api.ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == api_workflow_compute_worker.STATE_SCHEDULED_ID_NOT_FOUND
    assert run_info.message.startswith("Could not find a job for the given run_id:")
    assert run_info.in_end_state() == True


def test_get_compute_worker_state_and_message_docker_state() -> None:
    message = "SOME_MESSAGE"
    docker_run = models.DockerRunData(
        id=utils.generate_id(),
        user_id="user-id",
        state=models.DockerRunState.GENERATING_REPORT,
        docker_version="",
        created_at=0,
        last_modified_at=0,
        message=message,
    )
    mocked_api_client = mock.MagicMock(
        dataset_id=utils.generate_id(),
        _compute_worker_api=mock.MagicMock(
            get_docker_run_by_scheduled_id=lambda scheduled_id: docker_run
        ),
    )

    run_info = lightly_api.ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=utils.generate_id()
    )
    assert run_info.state == models.DockerRunState.GENERATING_REPORT
    assert run_info.message == message
    assert run_info.in_end_state() == False


def test_compute_worker_run_info_generator(mocker) -> None:
    states = [f"state_{i}" for i in range(7)]
    states[-1] = models.DockerRunState.COMPLETED

    class MockedApiWorkflowClient:
        def __init__(self, states: List[str]):
            self.states = states
            self.current_state_index = 0
            random.seed(42)

        def get_compute_worker_run_info(self, scheduled_run_id: str):
            state = self.states[self.current_state_index]
            if random.random() > 0.9:
                self.current_state_index += 1
            return api_workflow_compute_worker.ComputeWorkerRunInfo(
                state=state, message=state
            )

    mocker.patch("time.sleep", lambda _: None)

    mocked_client = MockedApiWorkflowClient(states)
    run_infos = list(
        lightly_api.ApiWorkflowClient.compute_worker_run_info_generator(
            mocked_client, scheduled_run_id=""
        )
    )

    expected_run_infos = [
        api_workflow_compute_worker.ComputeWorkerRunInfo(state=state, message=state)
        for state in states
    ]

    assert run_infos == expected_run_infos


def test_get_compute_worker_runs(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(lightly_api.ApiWorkflowClient, "__init__", return_value=None)
    dataset_id = utils.generate_id()
    run_ids = [utils.generate_id(), utils.generate_id()]
    client = lightly_api.ApiWorkflowClient(token="123")
    mock_compute_worker_api = mocker.create_autospec(
        swagger_api.DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_runs.side_effect = [
        [
            models.DockerRunData(
                id=run_ids[0],
                user_id="user-id",
                created_at=20,
                dataset_id=dataset_id,
                docker_version="",
                state=models.DockerRunState.COMPUTING_METADATA,
                last_modified_at=0,
            ),
            models.DockerRunData(
                id=run_ids[1],
                user_id="user-id",
                created_at=10,
                dataset_id=dataset_id,
                docker_version="",
                state=models.DockerRunState.COMPUTING_METADATA,
                last_modified_at=0,
            ),
        ],
        [],
    ]
    client._compute_worker_api = mock_compute_worker_api
    runs = client.get_compute_worker_runs()
    assert runs == [
        models.DockerRunData(
            id=run_ids[1],
            user_id="user-id",
            created_at=10,
            dataset_id=dataset_id,
            docker_version="",
            state=models.DockerRunState.COMPUTING_METADATA,
            last_modified_at=0,
        ),
        models.DockerRunData(
            id=run_ids[0],
            user_id="user-id",
            created_at=20,
            dataset_id=dataset_id,
            docker_version="",
            state=models.DockerRunState.COMPUTING_METADATA,
            last_modified_at=0,
        ),
    ]
    assert mock_compute_worker_api.get_docker_runs.call_count == 2


def test_get_compute_worker_runs__dataset(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch.object(lightly_api.ApiWorkflowClient, "__init__", return_value=None)
    dataset_id = utils.generate_id()
    run_id = utils.generate_id()
    client = lightly_api.ApiWorkflowClient(token="123")
    mock_compute_worker_api = mocker.create_autospec(
        swagger_api.DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_runs_query_by_dataset_id.side_effect = [
        [
            models.DockerRunData(
                id=run_id,
                user_id="user-id",
                dataset_id=dataset_id,
                docker_version="",
                state=models.DockerRunState.COMPUTING_METADATA,
                created_at=0,
                last_modified_at=0,
            ),
        ],
        [],
    ]

    client._compute_worker_api = mock_compute_worker_api
    runs = client.get_compute_worker_runs(dataset_id=dataset_id)
    assert runs == [
        models.DockerRunData(
            id=run_id,
            user_id="user-id",
            dataset_id=dataset_id,
            docker_version="",
            state=models.DockerRunState.COMPUTING_METADATA,
            created_at=0,
            last_modified_at=0,
        ),
    ]
    assert mock_compute_worker_api.get_docker_runs_query_by_dataset_id.call_count == 2


def test_get_compute_worker_run_tags__no_tags(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(lightly_api.ApiWorkflowClient, "__init__", return_value=None)
    run_id = utils.generate_id()
    client = lightly_api.ApiWorkflowClient(token="123", dataset_id=utils.generate_id())
    mock_compute_worker_api = mocker.create_autospec(
        swagger_api.DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_run_tags.return_value = []
    client._compute_worker_api = mock_compute_worker_api
    tags = client.get_compute_worker_run_tags(run_id=run_id)
    assert len(tags) == 0
    mock_compute_worker_api.get_docker_run_tags.assert_called_once_with(run_id=run_id)


def test_get_compute_worker_run_tags__single_tag(
    mocker: pytest_mock.MockerFixture,
) -> None:
    dataset_id = utils.generate_id()
    run_id = utils.generate_id()
    mocker.patch.object(lightly_api.ApiWorkflowClient, "__init__", return_value=None)
    client = lightly_api.ApiWorkflowClient(token="123", dataset_id=dataset_id)
    client._dataset_id = dataset_id
    mock_compute_worker_api = mocker.create_autospec(
        swagger_api.DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_run_tags.return_value = [
        models.TagData(
            id=utils.generate_id(),
            dataset_id=dataset_id,
            prev_tag_id=None,
            bit_mask_data="0x1",
            name="tag-0",
            tot_size=0,
            created_at=0,
            changes=None,
            run_id=run_id,
        )
    ]
    client._compute_worker_api = mock_compute_worker_api
    tags = client.get_compute_worker_run_tags(run_id=run_id)
    assert len(tags) == 1
    mock_compute_worker_api.get_docker_run_tags.assert_called_once_with(run_id=run_id)


def test_get_compute_worker_run_tags__multiple_tags(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(lightly_api.ApiWorkflowClient, "__init__", return_value=None)
    run_id = utils.generate_id()
    dataset_id = utils.generate_id()
    client = lightly_api.ApiWorkflowClient(token="123", dataset_id=dataset_id)
    client._dataset_id = dataset_id
    mock_compute_worker_api = mocker.create_autospec(
        swagger_api.DockerApi, spec_set=True
    ).return_value

    tag_ids = [utils.generate_id() for _ in range(3)]
    tag_0 = models.TagData(
        id=tag_ids[0],
        dataset_id=dataset_id,
        prev_tag_id=None,
        bit_mask_data="0x1",
        name="tag-0",
        tot_size=0,
        created_at=0,
        changes=None,
        run_id=run_id,
    )
    tag_1 = models.TagData(
        id=tag_ids[1],
        dataset_id=dataset_id,
        prev_tag_id=tag_ids[0],
        bit_mask_data="0x1",
        name="tag-1",
        tot_size=0,
        created_at=1,
        changes=None,
        run_id=run_id,
    )
    # tag from a different dataset
    tag_2 = models.TagData(
        id=tag_ids[2],
        dataset_id=utils.generate_id(),
        prev_tag_id=None,
        bit_mask_data="0x1",
        name="tag-2",
        tot_size=0,
        created_at=2,
        changes=None,
        run_id=run_id,
    )
    # tags are returned ordered by decreasing creation date
    mock_compute_worker_api.get_docker_run_tags.return_value = [tag_2, tag_1, tag_0]
    client._compute_worker_api = mock_compute_worker_api
    tags = client.get_compute_worker_run_tags(run_id="run-0")
    assert len(tags) == 2
    assert tags[0] == tag_1
    assert tags[1] == tag_0
    mock_compute_worker_api.get_docker_run_tags.assert_called_once_with(run_id="run-0")


def test__config_to_camel_case() -> None:
    assert api_workflow_compute_worker._config_to_camel_case(
        {
            "lorem_ipsum": "dolor",
            "lorem": {
                "ipsum_dolor": "sit_amet",
            },
        }
    ) == {
        "loremIpsum": "dolor",
        "lorem": {
            "ipsumDolor": "sit_amet",
        },
    }


def test__snake_to_camel_case() -> None:
    assert api_workflow_compute_worker._snake_to_camel_case("lorem") == "lorem"
    assert (
        api_workflow_compute_worker._snake_to_camel_case("lorem_ipsum") == "loremIpsum"
    )
    assert (
        api_workflow_compute_worker._snake_to_camel_case("lorem_ipsum_dolor")
        == "loremIpsumDolor"
    )
    assert (
        api_workflow_compute_worker._snake_to_camel_case("loremIpsum") == "loremIpsum"
    )  # do nothing


def test__validate_config__docker() -> None:
    obj = models.DockerWorkerConfigV3Docker(
        enable_training=False,
        corruptness_check=models.DockerWorkerConfigV3DockerCorruptnessCheck(
            corruption_threshold=0.1,
        ),
    )
    api_workflow_compute_worker._validate_config(
        cfg={
            "enable_training": False,
            "corruptness_check": {
                "corruption_threshold": 0.1,
            },
        },
        obj=obj,
    )


def test__validate_config__docker_typo() -> None:
    obj = models.DockerWorkerConfigV3Docker(
        enable_training=False,
        corruptness_check=models.DockerWorkerConfigV3DockerCorruptnessCheck(
            corruption_threshold=0.1,
        ),
    )

    with pytest.raises(
        api_workflow_compute_worker.InvalidConfigurationError,
        match="Option 'enable_trainingx' does not exist! Did you mean 'enable_training'?",
    ):
        api_workflow_compute_worker._validate_config(
            cfg={
                "enable_trainingx": False,
                "corruptness_check": {
                    "corruption_threshold": 0.1,
                },
            },
            obj=obj,
        )


def test__validate_config__docker_typo_nested() -> None:
    obj = models.DockerWorkerConfigV3Docker(
        enable_training=False,
        corruptness_check=models.DockerWorkerConfigV3DockerCorruptnessCheck(
            corruption_threshold=0.1,
        ),
    )

    with pytest.raises(
        api_workflow_compute_worker.InvalidConfigurationError,
        match="Option 'corruption_thresholdx' does not exist! Did you mean 'corruption_threshold'?",
    ):
        api_workflow_compute_worker._validate_config(
            cfg={
                "enable_training": False,
                "corruptness_check": {
                    "corruption_thresholdx": 0.1,
                },
            },
            obj=obj,
        )


def test__validate_config__lightly() -> None:
    obj = models.DockerWorkerConfigV3Lightly(
        loader=models.DockerWorkerConfigV3LightlyLoader(
            num_workers=-1,
            batch_size=16,
            shuffle=True,
        ),
        collate=models.DockerWorkerConfigV3LightlyCollate(
            rr_degrees=[-90, 90],
        ),
    )
    api_workflow_compute_worker._validate_config(
        cfg={
            "loader": {
                "num_workers": -1,
                "batch_size": 16,
                "shuffle": True,
            },
            "collate": {
                "rr_degrees": [-90, 90],
            },
        },
        obj=obj,
    )


def test__validate_config__lightly_typo() -> None:
    obj = models.DockerWorkerConfigV3Lightly(
        loader=models.DockerWorkerConfigV3LightlyLoader(
            num_workers=-1,
            batch_size=16,
            shuffle=True,
        )
    )
    with pytest.raises(
        api_workflow_compute_worker.InvalidConfigurationError,
        match="Option 'loaderx' does not exist! Did you mean 'loader'?",
    ):
        api_workflow_compute_worker._validate_config(
            cfg={
                "loaderx": {
                    "num_workers": -1,
                    "batch_size": 16,
                    "shuffle": True,
                },
            },
            obj=obj,
        )


def test__validate_config__lightly_typo_nested() -> None:
    obj = models.DockerWorkerConfigV3Lightly(
        loader=models.DockerWorkerConfigV3LightlyLoader(
            num_workers=-1,
            batch_size=16,
            shuffle=True,
        )
    )
    with pytest.raises(
        api_workflow_compute_worker.InvalidConfigurationError,
        match="Option 'num_workersx' does not exist! Did you mean 'num_workers'?",
    ):
        api_workflow_compute_worker._validate_config(
            cfg={
                "loader": {
                    "num_workersx": -1,
                    "batch_size": 16,
                    "shuffle": True,
                },
            },
            obj=obj,
        )
