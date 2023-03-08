import pickle

from lightly.api.swagger_api_client import LightlySwaggerApiClient
from lightly.openapi_generated.swagger_client import Configuration


def test_pickle() -> None:
    client = LightlySwaggerApiClient(configuration=Configuration(), timeout=5)
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
