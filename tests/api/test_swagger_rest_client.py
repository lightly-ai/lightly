import pickle

from pytest_mock import MockerFixture
from urllib3 import Timeout

from lightly.api import swagger_rest_client
from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_generated.swagger_client import Configuration


class TestLightlySwaggerRESTClientObject:
    def test__pickle(self) -> None:
        client = LightlySwaggerRESTClientObject(configuration=Configuration(), timeout=5)
        new_client = pickle.loads(pickle.dumps(client))

        assert set(client.__dict__.keys()) == set(new_client.__dict__.keys())
        assert all(
            type(client.__dict__[key]) == type(new_client.__dict__[key])
            for key in client.__dict__.keys()
        )

        original_dict = client.__dict__.copy()
        del original_dict["pool_manager"]  # different because pool_manager is recreated
        del original_dict["configuration"] # different because loggers inside configuration are recreated
        assert all(
            original_dict[key] == new_client.__dict__[key] for key in original_dict.keys()
        )

    def test_request__timeout(self, mocker: MockerFixture) -> None:
        client = LightlySwaggerRESTClientObject(configuration=Configuration(), timeout=5)
        response = mocker.MagicMock()
        response.status = 200
        client.pool_manager.request = mocker.MagicMock(return_value=response)

        # use default timeout
        client.request(method="GET", url="some-url")

        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[0]
        assert isinstance(kwargs['timeout'], Timeout)
        assert kwargs['timeout'].total == 5

        # use custom timeout
        client.request(method="GET", url="some-url", _request_timeout=10)
        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[1]
        assert isinstance(kwargs['timeout'], Timeout)
        assert kwargs['timeout'].total == 10

    def test_request__connection_read_timeout(self, mocker: MockerFixture) -> None:
        client = LightlySwaggerRESTClientObject(configuration=Configuration(), timeout=(1, 2))
        response = mocker.MagicMock()
        response.status = 200
        client.pool_manager.request = mocker.MagicMock(return_value=response)

        client.request(method="GET", url="some-url")
        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[0]
        assert isinstance(kwargs['timeout'], Timeout)
        assert kwargs['timeout'].connect_timeout == 1
        assert kwargs['timeout'].read_timeout == 2


    def test_request__flatten_list_query_parameters(self, mocker: MockerFixture) -> None:
        client = LightlySwaggerRESTClientObject(configuration=Configuration(), timeout=5)
        response = mocker.MagicMock()
        response.status = 200
        client.pool_manager.request = mocker.MagicMock(return_value=response)

        client.request(method="GET", url="some-url", query_params=[('param-name', ['value-1', 'value-2'])])
        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[0]
        assert kwargs["fields"] == [('param-name', 'value-1'), ('param-name', 'value-2')]



def test__flatten_list_query_parameters() -> None:
    params = swagger_rest_client._flatten_list_query_parameters(query_params=[('param-name', ['value-1', 'value-2'])])
    assert params == [('param-name', 'value-1'), ('param-name', 'value-2')]
