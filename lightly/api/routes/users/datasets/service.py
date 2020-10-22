""" Datasets Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from . import _prefix
import lightly.api.utils as utils


def put_image_type(dataset_id: str,
                   token: str,
                   img_type: str):
    """Adds the attribute imgType to the db dataset entry.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
        img_type:
            Whether the sample was fully uploaded (full), only a thumbnail
            (thumbnail) or only metadata (meta).

    Returns:
        A boolean value indicating a successful put request.

    Raises:
        RuntimeError if put was not successful.

    """
    dst_url = _prefix(dataset_id=dataset_id)
    payload = {
        'token': token
    }

    data = {
        'dataset': {
            'imgType': img_type
        }
    }

    response = utils.put_request(dst_url, json=data, params=payload)
    return response
