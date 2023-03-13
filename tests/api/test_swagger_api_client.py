import pickle

from pytest_mock import MockerFixture

from lightly.api.swagger_api_client import LightlySwaggerApiClient
from lightly.api.swagger_rest_client import LightlySwaggerRESTClientObject
from lightly.openapi_generated.swagger_client import Configuration
from lightly.openapi_generated.swagger_client.rest import RESTResponse


def test_pickle(mocker: MockerFixture) -> None:
    client = LightlySwaggerApiClient(configuration=Configuration(), timeout=5)
    client.last_response = mocker.MagicMock(spec_set=RESTResponse).return_value
    new_client = pickle.loads(pickle.dumps(client))

    expected = {
        "_pool": None,
        "client_side_validation": True,
        # "configuration", ignore because some parts of configuration are recreated on unpickling
        "cookie": None,
        "default_headers": {"User-Agent": "Swagger-Codegen/1.0.0/python"},
        # "last_response", ignore because it is not copied during pickling
        # "rest_client", ignore because some parts of rest client are recreated on unpickling
    }
    # Check that all expected values are set except the ignored ones.
    assert set(expected.keys()) == set(client.__dict__.keys()) - {
        "configuration",
        "last_response",
        "rest_client",
    }
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
