import pickle

from lightly.openapi_generated.swagger_client import Configuration


def test_make_swagger_configuration_picklable() -> None:
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
