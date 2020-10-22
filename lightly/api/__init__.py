""" The lightly.api module provides access to the Lightly web-app. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.api.routes.pip import get_version              # noqa: F401
from lightly.api.upload import upload_images_from_folder    # noqa: F401
from lightly.api.upload import upload_embeddings_from_csv   # noqa: F401
from lightly.api.upload import upload_file_with_signed_url  # noqa: F401
from lightly.api.download import get_samples_by_tag         # noqa: F401
