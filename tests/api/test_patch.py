import pickle
from lightly.api import api_workflow_client
from lightly.api import ApiWorkflowClient
from pytest_mock import MockerFixture
from lightly.openapi_generated.swagger_client import ApiClient, Configuration
from lightly.openapi_generated.swagger_client.rest import RESTClientObject


def test_make_swagger_generated_classes_picklable__api_workflow_client(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(api_workflow_client, "is_compatible_version", return_value=True)
    client = ApiWorkflowClient(token="123")
    client._dataset_id = "my-dataset-id"
    new_client = pickle.loads(pickle.dumps(client))

    assert set(client.__dict__.keys()) == set(new_client.__dict__.keys())
    assert all(
        type(client.__dict__[key]) == type(new_client.__dict__[key])
        for key in client.__dict__.keys()
    )

    # Cannot assert that all attributes have same values because equality for mixin
    # classes and api client does not hold. Only comparing selected attributes.
    assert client.token == new_client.token
    assert client._dataset_id == new_client._dataset_id == "my-dataset-id"
    assert client._creator == new_client._creator


def test_make_swagger_generated_classes_picklable__api_client() -> None:
    config = Configuration()
    client = ApiClient(configuration=config)
    new_client = pickle.loads(pickle.dumps(client))

    assert set(client.__dict__.keys()) == set(new_client.__dict__.keys())
    assert all(
        type(client.__dict__[key]) == type(new_client.__dict__[key])
        for key in client.__dict__.keys()
    )

    original_dict = client.__dict__.copy()
    del original_dict["configuration"]  # different because loggers are recreated
    del original_dict["rest_client"]  # different because new rest client is recreated
    assert all(
        original_dict[key] == new_client.__dict__[key] for key in original_dict.keys()
    )


def test_make_swagger_generated_classes_picklable__configuration() -> None:
    config = Configuration()
    new_config = pickle.loads(pickle.dumps(config))

    assert set(config.__dict__.keys()) == set(new_config.__dict__.keys())
    assert all(
        type(config.__dict__[key]) == type(new_config.__dict__[key])
        for key in config.__dict__.keys()
    )

    original_dict = config.__dict__.copy()
    del original_dict["logger_stream_handler"]  # new object created on unpickle
    del original_dict["logger_formatter"]  # new object created on unpickle
    assert all(
        original_dict[key] == new_config.__dict__[key] for key in original_dict.keys()
    )


def test_make_swagger_generated_classes_picklable__rest_client() -> None:
    config = Configuration()
    client = RESTClientObject(configuration=config)
    new_client = pickle.loads(pickle.dumps(client))

    # Empty set because "pool_manager" is the only attribute stored on the client and
    # it is removed by `RESTClientObject.__getstate__` during pickling.
    assert set(new_client.__dict__.keys()) == set()
