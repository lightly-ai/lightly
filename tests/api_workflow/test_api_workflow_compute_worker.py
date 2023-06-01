import json
import random
from typing import Any, List
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_compute_worker
from lightly.api.api_workflow_compute_worker import (
    STATE_SCHEDULED_ID_NOT_FOUND,
    ComputeWorkerRunInfo,
    InvalidConfigurationError,
    _config_to_camel_case,
    _snake_to_camel_case,
    _validate_config,
)
from lightly.openapi_generated.swagger_client.api import DockerApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.models import (
    DockerRunData,
    DockerRunScheduledData,
    DockerRunScheduledPriority,
    DockerRunScheduledState,
    DockerRunState,
    DockerWorkerConfig,
    DockerWorkerConfigV3Docker,
    DockerWorkerConfigV3DockerCorruptnessCheck,
    DockerWorkerConfigV3Lightly,
    DockerWorkerConfigV3LightlyCollate,
    DockerWorkerConfigV3LightlyLoader,
    DockerWorkerState,
    DockerWorkerType,
    SelectionConfig,
    SelectionConfigEntry,
    SelectionConfigEntryInput,
    SelectionConfigEntryStrategy,
    SelectionInputPredictionsName,
    SelectionInputType,
    SelectionStrategyThresholdOperation,
    SelectionStrategyType,
    TagData,
)
from lightly.openapi_generated.swagger_client.rest import ApiException
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from tests.api_workflow.utils import generate_id


class TestApiWorkflowComputeWorker(MockedApiWorkflowSetup):
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
                            "dataset_id": generate_id(),
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
            selection_config=SelectionConfig(
                n_samples=20,
                strategies=[
                    SelectionConfigEntry(
                        input=SelectionConfigEntryInput(
                            type=SelectionInputType.EMBEDDINGS,
                            dataset_id=generate_id(),
                            tag_name="some-tag-name",
                        ),
                        strategy=SelectionConfigEntryStrategy(
                            type=SelectionStrategyType.SIMILARITY,
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
            priority=DockerRunScheduledPriority.HIGH,
        )
        assert scheduled_run_id

    def test_schedule_compute_worker_run__runs_on(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={}, lightly_config={}, runs_on=[generate_id(), generate_id()]
        )
        assert scheduled_run_id

    def test_get_compute_worker_ids(self):
        ids = self.api_workflow_client.get_compute_worker_ids()
        assert all(isinstance(id_, str) for id_ in ids)

    def test_get_compute_workers(self):
        workers = self.api_workflow_client.get_compute_workers()
        assert len(workers) == 1
        assert workers[0].name == "worker-name-1"
        assert workers[0].state == DockerWorkerState.OFFLINE
        assert workers[0].labels == ["label-1"]

    def test_get_compute_worker_runs(self):
        runs = self.api_workflow_client.get_compute_worker_runs()
        assert len(runs) > 0
        assert all(isinstance(run, DockerRunData) for run in runs)

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
        api_client = ApiClient()

        obj_as_json = json.dumps(api_client.sanitize_for_serialization(obj))

        mocked_response = mock.MagicMock()
        mocked_response.data = obj_as_json
        obj_api = api_client.deserialize(mocked_response, type(obj).__name__)

        self.assertDictEqual(obj.to_dict(), obj_api.to_dict())

        return obj_api

    def xtest_selection_config(self):
        selection_config = SelectionConfig(
            n_samples=1,
            strategies=[
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(type=SelectionInputType.EMBEDDINGS),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.DIVERSITY,
                        stopping_condition_minimum_distance=-1,
                    ),
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(
                        type=SelectionInputType.SCORES,
                        task="my-classification-task",
                        score="uncertainty_margin",
                    ),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.WEIGHTS
                    ),
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(
                        type=SelectionInputType.METADATA, key="lightly.sharpness"
                    ),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.THRESHOLD,
                        threshold=20,
                        operation=SelectionStrategyThresholdOperation.BIGGER_EQUAL,
                    ),
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(
                        type=SelectionInputType.PREDICTIONS,
                        task="my_object_detection_task",
                        name=SelectionInputPredictionsName.CLASS_DISTRIBUTION,
                    ),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.BALANCE,
                        target={"Ambulance": 0.2, "Bus": 0.4},
                    ),
                ),
            ],
        )
        config = DockerWorkerConfig(
            worker_type=DockerWorkerType.FULL, selection=selection_config
        )

        self._check_if_openapi_generated_obj_is_valid(config)


def test_selection_config_from_dict() -> None:
    dataset_id = generate_id()
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
        ValidationError,
        match=r"strategies\n  ensure this value has at least 1 items",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_key() -> None:
    cfg = {"strategies": [], "invalid-key": 0}
    with pytest.raises(
        ValidationError,
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
        ValidationError,
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
        ValidationError,
        match=r"stoppingConditionMinimumDistance\n  extra fields not permitted",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_get_scheduled_run_by_id() -> None:
    run_ids = [generate_id() for _ in range(3)]
    scheduled_runs = [
        DockerRunScheduledData(
            id=run_id,
            dataset_id=generate_id(),
            config_id=generate_id(),
            priority=DockerRunScheduledPriority.MID,
            state=DockerRunScheduledState.OPEN,
            created_at=0,
            last_modified_at=1,
            runs_on=[],
        )
        for run_id in run_ids
    ]
    mocked_compute_worker_api = MagicMock(
        get_docker_runs_scheduled_by_dataset_id=lambda dataset_id: scheduled_runs
    )
    mocked_api_client = MagicMock(
        dataset_id="asdf", _compute_worker_api=mocked_compute_worker_api
    )

    scheduled_run_id = run_ids[2]
    scheduled_run_data = ApiWorkflowClient._get_scheduled_run_by_id(
        self=mocked_api_client, scheduled_run_id=scheduled_run_id
    )
    assert scheduled_run_data.id == scheduled_run_id


def test_get_scheduled_run_by_id_not_found() -> None:
    scheduled_runs = [
        DockerRunScheduledData(
            id=generate_id(),
            dataset_id=generate_id(),
            config_id=generate_id(),
            priority=DockerRunScheduledPriority.LOW,
            state=DockerRunScheduledState.OPEN,
            created_at=0,
            last_modified_at=1,
            runs_on=[],
        )
        for _ in range(3)
    ]
    mocked_compute_worker_api = MagicMock(
        get_docker_runs_scheduled_by_dataset_id=lambda dataset_id: scheduled_runs
    )
    mocked_api_client = MagicMock(
        dataset_id="asdf", _compute_worker_api=mocked_compute_worker_api
    )

    scheduled_run_id = "id_5"
    with pytest.raises(
        ApiException,
        match=f"No scheduled run found for run with scheduled_run_id='{scheduled_run_id}'.",
    ):
        ApiWorkflowClient._get_scheduled_run_by_id(
            self=mocked_api_client, scheduled_run_id=scheduled_run_id
        )


def test_get_compute_worker_state_and_message_OPEN() -> None:
    dataset_id = generate_id()
    scheduled_run = DockerRunScheduledData(
        id=generate_id(),
        dataset_id=dataset_id,
        config_id=generate_id(),
        priority=DockerRunScheduledPriority.MID,
        state=DockerRunScheduledState.OPEN,
        created_at=0,
        last_modified_at=1,
        runs_on=["worker-label"],
    )

    def mocked_raise_exception(*args, **kwargs):
        raise ApiException

    mocked_api_client = MagicMock(
        dataset_id=dataset_id,
        _compute_worker_api=MagicMock(
            get_docker_run_by_scheduled_id=mocked_raise_exception
        ),
        _get_scheduled_run_by_id=lambda id: scheduled_run,
    )

    run_info = ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == DockerRunScheduledState.OPEN
    assert run_info.message.startswith("Waiting for pickup by Lightly Worker.")
    assert run_info.in_end_state() == False


def test_get_compute_worker_state_and_message_CANCELED() -> None:
    def mocked_raise_exception(*args, **kwargs):
        raise ApiException

    mocked_api_client = MagicMock(
        dataset_id=generate_id(),
        _compute_worker_api=MagicMock(
            get_docker_run_by_scheduled_id=mocked_raise_exception
        ),
        _get_scheduled_run_by_id=mocked_raise_exception,
    )
    run_info = ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == STATE_SCHEDULED_ID_NOT_FOUND
    assert run_info.message.startswith("Could not find a job for the given run_id:")
    assert run_info.in_end_state() == True


def test_get_compute_worker_state_and_message_docker_state() -> None:
    message = "SOME_MESSAGE"
    docker_run = DockerRunData(
        id=generate_id(),
        user_id="user-id",
        state=DockerRunState.GENERATING_REPORT,
        docker_version="",
        created_at=0,
        last_modified_at=0,
        message=message,
    )
    mocked_api_client = MagicMock(
        dataset_id=generate_id(),
        _compute_worker_api=MagicMock(
            get_docker_run_by_scheduled_id=lambda scheduled_id: docker_run
        ),
    )

    run_info = ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=generate_id()
    )
    assert run_info.state == DockerRunState.GENERATING_REPORT
    assert run_info.message == message
    assert run_info.in_end_state() == False


def test_compute_worker_run_info_generator(mocker) -> None:
    states = [f"state_{i}" for i in range(7)]
    states[-1] = DockerRunState.COMPLETED

    class MockedApiWorkflowClient:
        def __init__(self, states: List[str]):
            self.states = states
            self.current_state_index = 0
            random.seed(42)

        def get_compute_worker_run_info(self, scheduled_run_id: str):
            state = self.states[self.current_state_index]
            if random.random() > 0.9:
                self.current_state_index += 1
            return ComputeWorkerRunInfo(state=state, message=state)

    mocker.patch("time.sleep", lambda _: None)

    mocked_client = MockedApiWorkflowClient(states)
    run_infos = list(
        ApiWorkflowClient.compute_worker_run_info_generator(
            mocked_client, scheduled_run_id=""
        )
    )

    expected_run_infos = [
        ComputeWorkerRunInfo(state=state, message=state) for state in states
    ]

    assert run_infos == expected_run_infos


def test_get_compute_worker_runs(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    run_ids = [generate_id(), generate_id()]
    client = ApiWorkflowClient(token="123")
    mock_compute_worker_api = mocker.create_autospec(
        DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_runs.side_effect = [
        [
            DockerRunData(
                id=run_ids[0],
                user_id="user-id",
                created_at=20,
                dataset_id=dataset_id,
                docker_version="",
                state=DockerRunState.COMPUTING_METADATA,
                last_modified_at=0,
            ),
            DockerRunData(
                id=run_ids[1],
                user_id="user-id",
                created_at=10,
                dataset_id=dataset_id,
                docker_version="",
                state=DockerRunState.COMPUTING_METADATA,
                last_modified_at=0,
            ),
        ],
        [],
    ]
    client._compute_worker_api = mock_compute_worker_api
    runs = client.get_compute_worker_runs()
    assert runs == [
        DockerRunData(
            id=run_ids[1],
            user_id="user-id",
            created_at=10,
            dataset_id=dataset_id,
            docker_version="",
            state=DockerRunState.COMPUTING_METADATA,
            last_modified_at=0,
        ),
        DockerRunData(
            id=run_ids[0],
            user_id="user-id",
            created_at=20,
            dataset_id=dataset_id,
            docker_version="",
            state=DockerRunState.COMPUTING_METADATA,
            last_modified_at=0,
        ),
    ]
    assert mock_compute_worker_api.get_docker_runs.call_count == 2


def test_get_compute_worker_runs__dataset(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    run_id = generate_id()
    client = ApiWorkflowClient(token="123")
    mock_compute_worker_api = mocker.create_autospec(
        DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_runs_query_by_dataset_id.side_effect = [
        [
            DockerRunData(
                id=run_id,
                user_id="user-id",
                dataset_id=dataset_id,
                docker_version="",
                state=DockerRunState.COMPUTING_METADATA,
                created_at=0,
                last_modified_at=0,
            ),
        ],
        [],
    ]

    client._compute_worker_api = mock_compute_worker_api
    runs = client.get_compute_worker_runs(dataset_id=dataset_id)
    assert runs == [
        DockerRunData(
            id=run_id,
            user_id="user-id",
            dataset_id=dataset_id,
            docker_version="",
            state=DockerRunState.COMPUTING_METADATA,
            created_at=0,
            last_modified_at=0,
        ),
    ]
    assert mock_compute_worker_api.get_docker_runs_query_by_dataset_id.call_count == 2


def test_get_compute_worker_run_tags__no_tags(mocker: MockerFixture) -> None:
    run_id = generate_id()
    client = ApiWorkflowClient(token="123", dataset_id=generate_id())
    mock_compute_worker_api = mocker.create_autospec(
        DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_run_tags.return_value = []
    client._compute_worker_api = mock_compute_worker_api
    tags = client.get_compute_worker_run_tags(run_id=run_id)
    assert len(tags) == 0
    mock_compute_worker_api.get_docker_run_tags.assert_called_once_with(run_id=run_id)


def test_get_compute_worker_run_tags__single_tag(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    run_id = generate_id()
    client = ApiWorkflowClient(token="123", dataset_id=dataset_id)
    mock_compute_worker_api = mocker.create_autospec(
        DockerApi, spec_set=True
    ).return_value
    mock_compute_worker_api.get_docker_run_tags.return_value = [
        TagData(
            id=generate_id(),
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


def test_get_compute_worker_run_tags__multiple_tags(mocker: MockerFixture) -> None:
    run_id = generate_id()
    dataset_id = generate_id()
    client = ApiWorkflowClient(token="123", dataset_id=dataset_id)
    mock_compute_worker_api = mocker.create_autospec(
        DockerApi, spec_set=True
    ).return_value

    tag_ids = [generate_id() for _ in range(3)]
    tag_0 = TagData(
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
    tag_1 = TagData(
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
    tag_2 = TagData(
        id=tag_ids[2],
        dataset_id=generate_id(),
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
    assert _config_to_camel_case(
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
    assert _snake_to_camel_case("lorem") == "lorem"
    assert _snake_to_camel_case("lorem_ipsum") == "loremIpsum"
    assert _snake_to_camel_case("lorem_ipsum_dolor") == "loremIpsumDolor"
    assert _snake_to_camel_case("loremIpsum") == "loremIpsum"  # do nothing


def test__validate_config__docker() -> None:
    obj = DockerWorkerConfigV3Docker(
        enable_training=False,
        corruptness_check=DockerWorkerConfigV3DockerCorruptnessCheck(
            corruption_threshold=0.1,
        ),
    )
    _validate_config(
        cfg={
            "enable_training": False,
            "corruptness_check": {
                "corruption_threshold": 0.1,
            },
        },
        obj=obj,
    )


def test__validate_config__docker_typo() -> None:
    obj = DockerWorkerConfigV3Docker(
        enable_training=False,
        corruptness_check=DockerWorkerConfigV3DockerCorruptnessCheck(
            corruption_threshold=0.1,
        ),
    )

    with pytest.raises(
        InvalidConfigurationError,
        match="Option 'enable_trainingx' does not exist! Did you mean 'enable_training'?",
    ):
        _validate_config(
            cfg={
                "enable_trainingx": False,
                "corruptness_check": {
                    "corruption_threshold": 0.1,
                },
            },
            obj=obj,
        )


def test__validate_config__docker_typo_nested() -> None:
    obj = DockerWorkerConfigV3Docker(
        enable_training=False,
        corruptness_check=DockerWorkerConfigV3DockerCorruptnessCheck(
            corruption_threshold=0.1,
        ),
    )

    with pytest.raises(
        InvalidConfigurationError,
        match="Option 'corruption_thresholdx' does not exist! Did you mean 'corruption_threshold'?",
    ):
        _validate_config(
            cfg={
                "enable_training": False,
                "corruptness_check": {
                    "corruption_thresholdx": 0.1,
                },
            },
            obj=obj,
        )


def test__validate_config__lightly() -> None:
    obj = DockerWorkerConfigV3Lightly(
        loader=DockerWorkerConfigV3LightlyLoader(
            num_workers=-1,
            batch_size=16,
            shuffle=True,
        ),
        collate=DockerWorkerConfigV3LightlyCollate(
            rr_degrees=[-90, 90],
        ),
    )
    _validate_config(
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
    obj = DockerWorkerConfigV3Lightly(
        loader=DockerWorkerConfigV3LightlyLoader(
            num_workers=-1,
            batch_size=16,
            shuffle=True,
        )
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="Option 'loaderx' does not exist! Did you mean 'loader'?",
    ):
        _validate_config(
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
    obj = DockerWorkerConfigV3Lightly(
        loader=DockerWorkerConfigV3LightlyLoader(
            num_workers=-1,
            batch_size=16,
            shuffle=True,
        )
    )
    with pytest.raises(
        InvalidConfigurationError,
        match="Option 'num_workersx' does not exist! Did you mean 'num_workers'?",
    ):
        _validate_config(
            cfg={
                "loader": {
                    "num_workersx": -1,
                    "batch_size": 16,
                    "shuffle": True,
                },
            },
            obj=obj,
        )
