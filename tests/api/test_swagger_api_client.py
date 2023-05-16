import pickle

from pytest_mock import MockerFixture

from lightly.api.swagger_api_client import (
    LightlySwaggerApiClient,
    _flatten_list_query_parameters,
)
from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_client import Configuration
from lightly.openapi_client.rest import RESTResponse


def test_pickle(mocker: MockerFixture) -> None:
    client = LightlySwaggerApiClient(configuration=Configuration(), timeout=5)
    client.last_response = mocker.MagicMock(spec_set=RESTResponse).return_value
    new_client = pickle.loads(pickle.dumps(client))

    expected = {
        "_pool": None,
        "client_side_validation": True,
        # "configuration", ignore because some parts of configuration are recreated on unpickling
        "cookie": None,
        "default_headers": {"User-Agent": "OpenAPI-Generator/1.0.0/python"},
        # "last_response", ignore because it is not copied during pickling
        # "rest_client", ignore because some parts of rest client are recreated on unpickling
    }
    # Check that all expected values are set except the ignored ones.
    assert all(hasattr(client, key) for key in expected.keys())
    # Check that new client values are equal to expected values.
    assert all(new_client.__dict__[key] == value for key, value in expected.items())

    # Extra assertions for attributes ignored in the tests above.
    assert isinstance(new_client.__dict__["configuration"], Configuration)
    assert isinstance(
        new_client.__dict__["rest_client"], LightlySwaggerRESTClientObject
    )
    # Last reponse is completely removed from client object and is only dynamically
    # reassigned in the ApiClient.__call_api method.
    assert not hasattr(new_client, "last_response")


def test__flatten_list_query_parameters() -> None:
    params = _flatten_list_query_parameters(
        query_params=[("param-name", ["value-1", "value-2"])]
    )
    assert params == [("param-name", "value-1"), ("param-name", "value-2")]
