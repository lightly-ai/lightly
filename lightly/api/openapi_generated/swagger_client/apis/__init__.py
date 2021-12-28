
# flake8: noqa

# Import all APIs into this package.
# If you have many APIs here with many many models used in each API this may
# raise a `RecursionError`.
# In order to avoid this, import only the API that you directly need like:
#
#
# or import this package, but before doing it, use:
#
#   import sys
#   sys.setrecursionlimit(n)

# Import APIs into API package:
from lightly.api.openapi_generated.swagger_client.api.datasets_api import DatasetsApi
from lightly.api.openapi_generated.swagger_client.api.datasources_api import DatasourcesApi
from lightly.api.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.api.openapi_generated.swagger_client.api.embeddings2d_api import Embeddings2dApi
from lightly.api.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.api.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.api.openapi_generated.swagger_client.api.meta_data_configurations_api import MetaDataConfigurationsApi
from lightly.api.openapi_generated.swagger_client.api.other_api import OtherApi
from lightly.api.openapi_generated.swagger_client.api.quota_api import QuotaApi
from lightly.api.openapi_generated.swagger_client.api.samples_api import SamplesApi
from lightly.api.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.api.openapi_generated.swagger_client.api.scores_api import ScoresApi
from lightly.api.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.api.openapi_generated.swagger_client.api.versioning_api import VersioningApi
