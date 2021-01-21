# openapi_client.TagsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_tag_by_dataset_id**](TagsApi.md#create_tag_by_dataset_id) | **POST** /users/datasets/{datasetId}/tags | create new tag for dataset
[**get_tags_by_dataset_id**](TagsApi.md#get_tags_by_dataset_id) | **GET** /users/datasets/{datasetId}/tags | Get all tags of a dataset
[**trigger_sampling_by_id**](TagsApi.md#trigger_sampling_by_id) | **POST** /v1/datasets/{datasetId}/tags/{tagId}/embeddings/{embeddingId}/sampling | Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding


# **create_tag_by_dataset_id**
> [tag_data.TagData] create_tag_by_dataset_id(dataset_id, inline_object1_inline_object1)

create new tag for dataset

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import lightly.openapi_generated.openapi_client
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): auth0Bearer
configuration = openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.TagsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    inline_object1_inline_object1 = openapi_client.InlineObject1() # inline_object1.InlineObject1 | 
    
    # example passing only required values which don't have defaults set
    try:
        # create new tag for dataset
        api_response = api_instance.create_tag_by_dataset_id(dataset_id, inline_object1_inline_object1)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling TagsApi->create_tag_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **inline_object1_inline_object1** | [**inline_object1.InlineObject1**](InlineObject1.md)|  |

### Return type

[**[tag_data.TagData]**](TagData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_tags_by_dataset_id**
> [tag_data.TagData] get_tags_by_dataset_id(dataset_id)

Get all tags of a dataset

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import lightly.openapi_generated.openapi_client
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): auth0Bearer
configuration = openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.TagsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    
    # example passing only required values which don't have defaults set
    try:
        # Get all tags of a dataset
        api_response = api_instance.get_tags_by_dataset_id(dataset_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling TagsApi->get_tags_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |

### Return type

[**[tag_data.TagData]**](TagData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **trigger_sampling_by_id**
> inline_response2002.InlineResponse2002 trigger_sampling_by_id(dataset_id, tag_id, embedding_id, sampling_create_request_sampling_create_request)

Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import lightly.openapi_generated.openapi_client
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): auth0Bearer
configuration = openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.TagsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    tag_id = 'tag_id_example' # str | ObjectId of the tag
    embedding_id = 'embedding_id_example' # str | ObjectId of the prev uploaded embedding
    sampling_create_request_sampling_create_request = openapi_client.SamplingCreateRequest() # sampling_create_request.SamplingCreateRequest | 
    
    # example passing only required values which don't have defaults set
    try:
        # Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding
        api_response = api_instance.trigger_sampling_by_id(dataset_id, tag_id, embedding_id, sampling_create_request_sampling_create_request)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling TagsApi->trigger_sampling_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **tag_id** | **str**| ObjectId of the tag |
 **embedding_id** | **str**| ObjectId of the prev uploaded embedding |
 **sampling_create_request_sampling_create_request** | [**sampling_create_request.SamplingCreateRequest**](SamplingCreateRequest.md)|  |

### Return type

[**inline_response2002.InlineResponse2002**](InlineResponse2002.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

