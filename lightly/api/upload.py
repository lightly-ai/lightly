from lightly.api.utils import put_request



def upload_file_with_signed_url(file, url: str) -> bool:
    """Upload a file to the cloud storage using a signed URL.

    Args:
        file:
            The buffered file reader to upload.
        url:
            Signed url for push.

    Returns:
        A boolean value indicating successful upload.

    Raises:
        RuntimeError if put request failed.
    """
    response = put_request(url, data=file)
    file.close()
    return response
