import json
import logging
import time

import pytest
import requests
from pytest_mock import MockerFixture
from requests import Session
from requests.exceptions import RequestException
from urllib3.exceptions import ProtocolError
from urllib3.response import HTTPResponse

from lightly.api import retry_utils
from lightly.api.retry_utils import (
    DatasetNotFoundError,
    MaxRetryError,
    RetryConfig,
    RetryOnApiConfig,
)
from lightly.openapi_generated.swagger_client.exceptions import ApiException
from lightly.openapi_generated.swagger_client.models.api_error_code import ApiErrorCode


def test_retry_on_requests_error__retry(mocker: MockerFixture) -> None:
    # Skip waiting in retry backoff
    mocker.patch.object(time, "sleep")

    def load_url() -> None:
        raise ProtocolError()

    with pytest.raises(MaxRetryError) as exception:
        retry_utils.retry_on_requests_error(load_url)
    assert "5 attempt(s)" in str(exception.value)


def test_retry_on_requests_error__propagate_request_exception() -> None:
    session = Session()

    def load_url(url: str, session: Session) -> None:
        session.get(url, stream=True, timeout=0.1)

    # Making an impossible request using a session with a retry configuration
    # should raise a RequestException from the requests module.
    with pytest.raises(RequestException):
        retry_utils.retry_on_requests_error(load_url, "non-existing-url", session)


def test_retry_on_requests_error__retry_api_exception(mocker: MockerFixture) -> None:
    # Skip waiting in retry backoff
    mocker.patch.object(time, "sleep")

    def load_url() -> None:
        raise ApiException(400)

    with pytest.raises(MaxRetryError):
        retry_utils.retry_on_requests_error(load_url)


def test_retry_on_requests_error__retry_requests_exception(
    mocker: MockerFixture,
) -> None:
    # Skip waiting in retry backoff
    mocker.patch.object(time, "sleep")

    # Test that we retry on default requests exceptions.
    def request_retry() -> None:
        raise RequestException()

    with pytest.raises(MaxRetryError):
        retry_utils.retry_on_requests_error(request_retry)

    # Test that we retry HTTP errors without a response.
    def request_retry_http_no_response() -> None:
        raise requests.exceptions.HTTPError()

    with pytest.raises(MaxRetryError):
        retry_utils.retry_on_requests_error(request_retry_http_no_response)

    # Test that we retry on certain HTTP errors.
    def request_retry_http_400() -> None:
        response = requests.Response()
        response.status_code = 400
        raise requests.exceptions.HTTPError("", response=response)

    with pytest.raises(MaxRetryError):
        retry_utils.retry_on_requests_error(request_retry_http_400)

    # Test that we do not retry on certain HTTP errors.
    def request_no_retry_http_404() -> None:
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError("", response=response)

    with pytest.raises(requests.exceptions.HTTPError):
        retry_utils.retry_on_requests_error(request_no_retry_http_404)

    # Test that a non-retryable exception is propagated.
    def request_no_retry() -> None:
        raise requests.exceptions.URLRequired()

    with pytest.raises(requests.exceptions.URLRequired):
        retry_utils.retry_on_requests_error(request_no_retry)


def test_retry_on_api_error_should_retry() -> None:
    retry = retry_utils.RetryOnApiError(config=RetryOnApiConfig())
    assert not retry.should_retry(ApiException(404))
    assert not retry.should_retry(ZeroDivisionError())
    assert retry.should_retry(ApiException(500))


def test_retry_format_error() -> None:
    class SomeException(Exception):
        def __str__(self) -> str:
            return "this is an error"

    retry = retry_utils.Retry(config=RetryConfig())
    assert (
        retry.format_error(SomeException())
        == "SomeException. Details: this is an error"
    )


def test_retry_not_retried() -> None:
    config = RetryOnApiConfig(max_retries=2, backoff_factor=0.0)
    retry = retry_utils.RetryOnApiError(config=config)

    def raise_404() -> None:
        raise ApiException(404)

    with pytest.raises(ApiException):
        retry(raise_404)


def test_retry_retried() -> None:
    config = RetryOnApiConfig(max_retries=2, backoff_factor=0.0)
    retry = retry_utils.RetryOnApiError(config=config)

    def raise_500(param_a: str, param_b: str = "b", param_c: str = "c") -> None:
        raise ApiException(status=500, reason="foobar")

    with pytest.raises(MaxRetryError) as exception:
        retry(raise_500, "a", param_c="d")
    assert str(exception.value) == (
        "Calling 'raise_500' failed after 2 attempt(s). "
        "Args: ('a',); kwargs: {'param_c': 'd'}. "
        "Last error: ApiException. "
        "Details: (500)\nReason: foobar\n"
    )


def test_retry_urllib3_error_retried() -> None:
    config = RetryOnApiConfig(max_retries=2, backoff_factor=0.0)
    retry = retry_utils.RetryOnApiError(config=config)

    def raise_protocol_error() -> None:
        raise ProtocolError()

    with pytest.raises(MaxRetryError) as exception:
        retry(raise_protocol_error)
    assert str(exception.value) == (
        "Calling 'raise_protocol_error' failed after 2 attempt(s). "
        "Args: (); kwargs: {}. "
        "Last error: ProtocolError. Details: None"
    )


def test_retry__wrap_exception__dataset_unknown() -> None:
    def get_dataset() -> None:
        raise ApiException(
            http_resp=HTTPResponse(
                body=json.dumps(
                    {
                        "code": ApiErrorCode.DATASET_UNKNOWN,
                        "error": "...",
                        "requestId": "...",
                    }
                )
            )
        )

    with pytest.raises(DatasetNotFoundError):
        retry_utils.retry(get_dataset)


class TestRetryOnApiError:
    def test_should_retry__api_error_code(self) -> None:
        unknown_sample_ex = ApiException(
            http_resp=HTTPResponse(
                body=json.dumps(
                    {
                        "code": ApiErrorCode.SAMPLE_UNKNOWN,
                        "error": "...",
                        "requestId": "...",
                    }
                )
            )
        )
        malformed_request_ex = ApiException(
            http_resp=HTTPResponse(
                body=json.dumps(
                    {
                        "code": ApiErrorCode.MALFORMED_REQUEST,
                        "error": "...",
                        "requestId": "...",
                    }
                )
            )
        )
        config = RetryOnApiConfig(retry_api_error_codes={ApiErrorCode.SAMPLE_UNKNOWN})
        retry = retry_utils.RetryOnApiError(config=config)
        assert retry.should_retry(unknown_sample_ex)
        assert not retry.should_retry(malformed_request_ex)


def test__get_error_code_from_api_exception() -> None:
    ex = ApiException(
        http_resp=HTTPResponse(
            body=json.dumps(
                {
                    "code": ApiErrorCode.SAMPLE_UNKNOWN,
                    "error": "...",
                    "requestId": "...",
                }
            )
        )
    )
    assert (
        retry_utils._get_error_code_from_api_exception(ex=ex)
        == ApiErrorCode.SAMPLE_UNKNOWN
    )


def test__get_error_code_from_api_exception__body_empty() -> None:
    assert retry_utils._get_error_code_from_api_exception(ex=ApiException()) is None


def test__get_error_code_from_api_exception__body_not_json() -> None:
    ex = ApiException(http_resp=HTTPResponse(body=b"123"))
    assert retry_utils._get_error_code_from_api_exception(ex=ex) is None


def test__get_error_code_from_api_exception__body_not_dict() -> None:
    ex = ApiException(http_resp=HTTPResponse(body=json.dumps("some-string")))
    assert retry_utils._get_error_code_from_api_exception(ex=ex) is None


def test__get_error_code_from_api_exception__body_no_code() -> None:
    ex = ApiException(
        http_resp=HTTPResponse(body=json.dumps({"error": "...", "requestId": "..."}))
    )
    assert retry_utils._get_error_code_from_api_exception(ex=ex) is None


# TODO(Guarin, 02/23): Disabled this test because we temporarily disabled logging
# from different processes. Enable again once we switch to using spawn or forkserver to
# start processes. See LIG-2486.
@pytest.mark.skip(reason="Disabled logging from different processes.")
def test_retry__prints_log(caplog: pytest.LogCaptureFixture) -> None:
    config = RetryOnApiConfig(max_retries=1, backoff_factor=0.0)
    retry = retry_utils.RetryOnApiError(config=config)

    first = True

    def first_raise_then_pass() -> None:
        nonlocal first
        if first:
            first = False
            raise ApiException(500)

    # Retry logs on debug level, which gets to log.txt
    with caplog.at_level(logging.DEBUG):
        retry(first_raise_then_pass)

    assert len(caplog.records) == 1
    assert "Retry 1 of 1" in caplog.text


def test_retry__wrap_exception__any_error() -> None:
    class SomeException(Exception):
        pass

    some_exception = SomeException()
    simple_retry = retry_utils.Retry(config=RetryConfig())
    new_exception = simple_retry._wrap_exception(some_exception)
    assert new_exception == some_exception
