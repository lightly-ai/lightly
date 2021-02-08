# lightly.openapi_generated_with_other_gen.openapi_client.TagsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_initial_tag_by_dataset_id**](TagsApi.md#create_initial_tag_by_dataset_id) | **POST** /v1/datasets/{datasetId}/tags/initial | 
[**create_tag_by_dataset_id**](TagsApi.md#create_tag_by_dataset_id) | **POST** /v1/datasets/{datasetId}/tags | 
[**get_filenames_by_tag_id**](TagsApi.md#get_filenames_by_tag_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId}/filenames | 
[**get_tag_by_tag_id**](TagsApi.md#get_tag_by_tag_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId} | 
[**get_tags_by_dataset_id**](TagsApi.md#get_tags_by_dataset_id) | **GET** /v1/datasets/{datasetId}/tags | 


# **create_initial_tag_by_dataset_id**
> CreateEntityResponse create_initial_tag_by_dataset_id(dataset_id, initial_tag_create_request)



create the intitial tag for a dataset which then locks the dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import tags_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.initial_tag_create_request import InitialTagCreateRequest
from lightly.openapi_generated_with_other_gen.openapi_client.model.create_entity_response import CreateEntityResponse
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ApiKeyAuth
configuration.api_key['ApiKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKeyAuth'] = 'Bearer'

# Configure Bearer authorization (JWT): auth0Bearer
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    initial_tag_create_request = InitialTagCreateRequest(
        name=TagName("initial-tag"),
    ) # InitialTagCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_initial_tag_by_dataset_id(dataset_id, initial_tag_create_request)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling TagsApi->create_initial_tag_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **initial_tag_create_request** | [**InitialTagCreateRequest**](InitialTagCreateRequest.md)|  |

### Return type

[**CreateEntityResponse**](CreateEntityResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_tag_by_dataset_id**
> CreateEntityResponse create_tag_by_dataset_id(dataset_id, tag_create_request)



create new tag for dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import tags_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.create_entity_response import CreateEntityResponse
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_create_request import TagCreateRequest
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ApiKeyAuth
configuration.api_key['ApiKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKeyAuth'] = 'Bearer'

# Configure Bearer authorization (JWT): auth0Bearer
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_create_request = TagCreateRequest(
        name=TagName("initial-tag"),
        prev_tag_id=MongoObjectID("50000000abcdef1234566789"),
        bit_mask_data=TagBitMaskData("0x80bda23e9"),
        tot_size=1,
        changes=TagChangeData(),
    ) # TagCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_tag_by_dataset_id(dataset_id, tag_create_request)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling TagsApi->create_tag_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_create_request** | [**TagCreateRequest**](TagCreateRequest.md)|  |

### Return type

[**CreateEntityResponse**](CreateEntityResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_filenames_by_tag_id**
> TagFilenamesData get_filenames_by_tag_id(dataset_id, tag_id)



Get list of filenames by tag

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import tags_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_filenames_data import TagFilenamesData
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ApiKeyAuth
configuration.api_key['ApiKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKeyAuth'] = 'Bearer'

# Configure Bearer authorization (JWT): auth0Bearer
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_filenames_by_tag_id(dataset_id, tag_id)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling TagsApi->get_filenames_by_tag_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |

### Return type

[**TagFilenamesData**](TagFilenamesData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_tag_by_tag_id**
> TagData get_tag_by_tag_id(dataset_id, tag_id)



Get information about a specific tag

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import tags_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_data import TagData
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ApiKeyAuth
configuration.api_key['ApiKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKeyAuth'] = 'Bearer'

# Configure Bearer authorization (JWT): auth0Bearer
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_tag_by_tag_id(dataset_id, tag_id)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling TagsApi->get_tag_by_tag_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |

### Return type

[**TagData**](TagData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_tags_by_dataset_id**
> [TagData] get_tags_by_dataset_id(dataset_id)



Get all tags of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import tags_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_data import TagData
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: ApiKeyAuth
configuration.api_key['ApiKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKeyAuth'] = 'Bearer'

# Configure Bearer authorization (JWT): auth0Bearer
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_tags_by_dataset_id(dataset_id)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling TagsApi->get_tags_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |

### Return type

[**[TagData]**](TagData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

