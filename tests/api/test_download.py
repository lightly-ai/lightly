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

    # Mock retry function to be a context manager

    # Use real path in tmp_path
    output_path = tmp_path / "subdir" / "output.jpg"

    # Call function
    download.download_and_write_file(
        url="http://example.com/file.jpg",
        output_path=str(output_path),
        retry_fn=lambda fn, *args, **kwargs: fn(*args, **kwargs),
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

    # Use real path in tmp_path
    output_path = tmp_path / "output.jpg"

    # Call function
    download.download_and_write_file(
        url="http://example.com/file.jpg",
        output_path=str(output_path),
        session=mock_session,
        retry_fn=lambda fn, *args, **kwargs: fn(*args, **kwargs),
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

    file_infos = [
        ("file1.jpg", "http://example.com/file1.jpg"),
        ("file2.jpg", "http://example.com/file2.jpg"),
    ]

    # Call function
    download.download_and_write_all_files(
        file_infos=file_infos,
        output_dir=str(tmp_path),
        retry_fn=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )

    # Verify download_and_write_file was called for each file
    assert mock_download_and_write_file.call_count == 2

    # Verify the calls were made with correct arguments using assert_has_calls
    expected_calls = [
        mocker.call(
            "http://example.com/file1.jpg",
            str(tmp_path / "file1.jpg"),
            mocker.ANY,  # session
            mocker.ANY,  # retry_fn
            None,  # request_kwargs
        ),
        mocker.call(
            "http://example.com/file2.jpg",
            str(tmp_path / "file2.jpg"),
            mocker.ANY,  # session
            mocker.ANY,  # retry_fn
            None,  # request_kwargs
        ),
    ]

    # Use assert_has_calls which properly handles mocker.ANY comparisons
    mock_download_and_write_file.assert_has_calls(expected_calls, any_order=True)
