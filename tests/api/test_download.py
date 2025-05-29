import pathlib
from unittest.mock import MagicMock, mock_open

from pytest_mock import MockerFixture

from lightly.api import download


def test_download_and_write_file(mocker: MockerFixture, tmp_path: pathlib.Path) -> None:
    # Mock dependencies
    mock_response = MagicMock()
    mock_response.raw = MagicMock()
    mock_response_manager = MagicMock()
    mock_response_manager.__enter__.return_value = mock_response
    mock_requests_get = mocker.patch("requests.get", return_value=mock_response_manager)
    mock_open_file = mocker.patch("builtins.open", mock_open())
    mock_shutil_copyfileobj = mocker.patch("shutil.copyfileobj")

    # Mock retry function
    mock_retry_fn = MagicMock(
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )

    # Use real path in tmp_path
    output_path = tmp_path / "subdir" / "output.jpg"

    # Call function
    download.download_and_write_file(
        url="http://example.com/file.jpg",
        output_path=str(output_path),
        retry_fn=mock_retry_fn,
    )

    # Verify calls
    mock_response.raise_for_status.assert_called_once()
    mock_shutil_copyfileobj.assert_called_once_with(
        mock_response.raw, mock_open_file.return_value.__enter__.return_value
    )


def test_download_and_write_file__with_session(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    # Mock dependencies
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.raw = MagicMock()
    mock_response_manager = MagicMock()
    mock_response_manager.__enter__.return_value = mock_response
    mock_session.get.return_value = mock_response_manager
    mock_open_file = mocker.patch("builtins.open", mock_open())
    mock_shutil_copyfileobj = mocker.patch("shutil.copyfileobj")

    # Mock retry function
    mock_retry_fn = MagicMock(
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )

    # Use real path in tmp_path
    output_path = tmp_path / "output.jpg"

    # Call function
    download.download_and_write_file(
        url="http://example.com/file.jpg",
        output_path=str(output_path),
        session=mock_session,
        retry_fn=mock_retry_fn,
    )

    # Verify session was used instead of requests
    mock_session.get.assert_called_once_with(
        url="http://example.com/file.jpg", stream=True, timeout=10
    )


def test_download_and_write_all_files(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    # Mock the download_and_write_file function
    mock_download_and_write_file = mocker.patch(
        "lightly.api.download.download_and_write_file"
    )

    # Mock retry function
    mock_retry_fn = MagicMock(
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )

    file_infos = [
        ("file1.jpg", "http://example.com/file1.jpg"),
        ("file2.jpg", "http://example.com/file2.jpg"),
    ]

    # Call function
    download.download_and_write_all_files(
        file_infos=file_infos,
        output_dir=str(tmp_path),
        retry_fn=mock_retry_fn,
    )

    # Verify download_and_write_file was called for each file
    assert mock_download_and_write_file.call_count == 2

    # Verify the calls were made with correct arguments
    expected_calls = [
        mocker.call(
            url="http://example.com/file1.jpg",
            output_path=str(tmp_path / "file1.jpg"),
            session=mocker.ANY,
            retry_fn=mock_retry_fn,
            request_kwargs=None,
        ),
        mocker.call(
            url="http://example.com/file2.jpg",
            output_path=str(tmp_path / "file2.jpg"),
            session=mocker.ANY,
            retry_fn=mock_retry_fn,
            request_kwargs=None,
        ),
    ]

    # Use assert_has_calls to verify the calls
    mock_download_and_write_file.assert_has_calls(expected_calls, any_order=True)
