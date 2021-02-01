from typing import *

from lightly.openapi_generated.swagger_client import ApiClient
from lightly.openapi_generated.swagger_client.models import TagData, TagName, TagChangeData, TagCreateRequest
from lightly.openapi_generated.swagger_client.api import TagsApi


def get_tags_by_dataset_id(api_client: ApiClient, dataset_id: str) -> List[TagData]:
    tags_api = TagsApi(api_client=api_client)
    tags = tags_api.get_tags_by_dataset_id(dataset_id=dataset_id)
    return tags


def get_tag_by_tag_id(api_client: ApiClient, dataset_id: str, tag_id: str) -> TagData:
    tag = TagsApi(api_client).get_tag_by_tag_id(dataset_id=dataset_id,tag_id=tag_id)
    return tag
