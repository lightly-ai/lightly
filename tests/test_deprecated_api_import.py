import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "import_statement",
    [
        "import lightly.api",
        "from lightly import api",
        "from lightly.api import ApiWorkflowClient",
    ],
)
def test_lightly_api_imports_raise_deprecation_error(import_statement: str) -> None:
    sys.modules.pop("lightly.api", None)

    with pytest.raises(ImportError) as error:
        exec(import_statement, {})

    message = str(error.value)
    assert "ApiWorkflowClient" in message
    assert "deprecated" in message
    assert "v1.15" in message
    assert "lightly<1.16" in message


def test_lightly_api_importlib_import_raises_deprecation_error() -> None:
    sys.modules.pop("lightly.api", None)

    with pytest.raises(ImportError, match="ApiWorkflowClient"):
        importlib.import_module("lightly.api")
