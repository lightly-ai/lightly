
# flake8: noqa

# Import all APIs into this package.
# If you have many APIs here with many many models used in each API this may
# raise a `RecursionError`.
# In order to avoid this, import only the API that you directly need like:
#
#   from .api.annotations_api import AnnotationsApi
#
# or import this package, but before doing it, use:
#
#   import sys
#   sys.setrecursionlimit(n)

# Import APIs into API package:
from lightly.openapi_generated_with_other_gen.openapi_client.api.annotations_api import AnnotationsApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.auth_api import AuthApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.datasets_api import DatasetsApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.internal_api import InternalApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.jobs_api import JobsApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.mappings_api import MappingsApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.samples_api import SamplesApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated_with_other_gen.openapi_client.api.tags_api import TagsApi
