import pickle

from pytest_mock import MockerFixture
from urllib3 import PoolManager, Timeout

from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_client.configuration import Configuration


class TestLightlySwaggerRESTClientObject:
    def test__pickle(self) -> None:
        client = LightlySwaggerRESTClientObject(
            configuration=Configuration(), timeout=5
        )
        new_client = pickle.loads(pickle.dumps(client))
        expected = {
            # "configuration", ignore because some parts of configuration are recreated on unpickling
            "maxsize": None,
            # "pool_manager", ignore because pool_manager is recreated on unpickling
            "pools_size": 4,
            "timeout": 5,
        }

        # Check that all expected values are set except the ignored ones.
        assert set(expected.keys()) == set(client.__dict__.keys()) - {
            "configuration",
            "pool_manager",
        }
        # Check that new client values are equal to expected values.
        assert all(new_client.__dict__[key] == value for key, value in expected.items())

        # Extra assertions for attributes ignored in the tests above.
        assert isinstance(new_client.__dict__["configuration"], Configuration)
        assert isinstance(new_client.__dict__["pool_manager"], PoolManager)

    def test_request__timeout(self, mocker: MockerFixture) -> None:
        client = LightlySwaggerRESTClientObject(
            configuration=Configuration(), timeout=5
        )
        response = mocker.MagicMock()
        response.status = 200
        client.pool_manager.request = mocker.MagicMock(return_value=response)

        # use default timeout
        client.request(method="GET", url="some-url")

        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[0]
        assert isinstance(kwargs["timeout"], Timeout)
        assert kwargs["timeout"].total == 5

        # use custom timeout
        client.request(method="GET", url="some-url", _request_timeout=10)
        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[1]
        assert isinstance(kwargs["timeout"], Timeout)
        assert kwargs["timeout"].total == 10

    def test_request__connection_read_timeout(self, mocker: MockerFixture) -> None:
        client = LightlySwaggerRESTClientObject(
            configuration=Configuration(), timeout=(1, 2)
        )
        response = mocker.MagicMock()
        response.status = 200
        client.pool_manager.request = mocker.MagicMock(return_value=response)

        client.request(method="GET", url="some-url")
        calls = client.pool_manager.request.mock_calls
        _, _, kwargs = calls[0]
        assert isinstance(kwargs["timeout"], Timeout)
        assert kwargs["timeout"].connect_timeout == 1
        assert kwargs["timeout"].read_timeout == 2
