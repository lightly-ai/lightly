# lightly.openapi_generated.swagger_client.TagsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_tag_by_dataset_id**](TagsApi.md#create_tag_by_dataset_id) | **POST** /users/datasets/{datasetId}/tags | create new tag for dataset
[**get_tags_by_dataset_id**](TagsApi.md#get_tags_by_dataset_id) | **GET** /users/datasets/{datasetId}/tags | Get all tags of a dataset
[**trigger_sampling_by_id**](TagsApi.md#trigger_sampling_by_id) | **POST** /v1/datasets/{datasetId}/tags/{tagId}/embeddings/{embeddingId}/sampling | Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

# **create_tag_by_dataset_id**
> list[TagData] create_tag_by_dataset_id(body, dataset_id)

create new tag for dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.TagsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
body = lightly.openapi_generated.swagger_client.Body1() # Body1 | 
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset

try:
    # create new tag for dataset
    api_response = api_instance.create_tag_by_dataset_id(body, dataset_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TagsApi->create_tag_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Body1**](Body1.md)|  | 
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 

### Return type

[**list[TagData]**](TagData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_tags_by_dataset_id**
> list[TagData] get_tags_by_dataset_id(dataset_id)

Get all tags of a dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.TagsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset

try:
    # Get all tags of a dataset
    api_response = api_instance.get_tags_by_dataset_id(dataset_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TagsApi->get_tags_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 

### Return type

[**list[TagData]**](TagData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **trigger_sampling_by_id**
> InlineResponse2003 trigger_sampling_by_id(body, dataset_id, tag_id, embedding_id)

Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.TagsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
body = lightly.openapi_generated.swagger_client.SamplingCreateRequest() # SamplingCreateRequest | 
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
tag_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the tag
embedding_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the prev uploaded embedding

try:
    # Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding
    api_response = api_instance.trigger_sampling_by_id(body, dataset_id, tag_id, embedding_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TagsApi->trigger_sampling_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**SamplingCreateRequest**](SamplingCreateRequest.md)|  | 
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **tag_id** | [**MongoObjectID**](.md)| ObjectId of the tag | 
 **embedding_id** | [**MongoObjectID**](.md)| ObjectId of the prev uploaded embedding | 

### Return type

[**InlineResponse2003**](InlineResponse2003.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

