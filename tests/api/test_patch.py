import logging
import pickle

from lightly.openapi_client import Configuration


def test_make_swagger_configuration_picklable() -> None:
    config = Configuration()
    # Fix value to make test reproducible on systems with different number of cpus.
    config.connection_pool_maxsize = 4
    new_config = pickle.loads(pickle.dumps(config))

    expected = {
        "_Configuration__debug": False,
        "_Configuration__logger_file": None,
        "_Configuration__logger_format": "%(asctime)s %(levelname)s %(message)s",
        "api_key_prefix": {},
        "api_key": {},
        "assert_hostname": None,
        "cert_file": None,
        "client_side_validation": True,
        "connection_pool_maxsize": 4,
        "_base_path": "https://api.lightly.ai",
        "key_file": None,
        "logger_file_handler": None,
        # "logger_formatter", ignore because a new object is created on unpickle
        # "logger_stream_handler", ignore because a new object is created on unpickle
        "logger": {
            "package_logger": logging.getLogger("lightly.openapi_client"),
            "urllib3_logger": logging.getLogger("urllib3"),
        },
        "password": None,
        "proxy": None,
        "refresh_api_key_hook": None,
        "safe_chars_for_path_param": "",
        "ssl_ca_cert": None,
        "temp_folder_path": None,
        "username": None,
        "verify_ssl": True,
    }
    # Check that all expected values are set except the ignored ones.
    assert all(hasattr(config, key) for key in expected.keys())
    # Check that new_config values are equal to expected values.
    assert all(new_config.__dict__[key] == value for key, value in expected.items())

    # Extra assertions for attributes ignored in the tests above.
    assert isinstance(new_config.__dict__["logger_formatter"], logging.Formatter)
    assert isinstance(
        new_config.__dict__["logger_stream_handler"], logging.StreamHandler
    )
