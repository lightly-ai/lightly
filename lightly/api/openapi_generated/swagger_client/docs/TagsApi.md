# swagger_client.TagsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_initial_tag_by_dataset_id**](TagsApi.md#create_initial_tag_by_dataset_id) | **POST** /v1/datasets/{datasetId}/tags/initial | 
[**create_tag_by_dataset_id**](TagsApi.md#create_tag_by_dataset_id) | **POST** /v1/datasets/{datasetId}/tags | 
[**delete_tag_by_tag_id**](TagsApi.md#delete_tag_by_tag_id) | **DELETE** /v1/datasets/{datasetId}/tags/{tagId} | 
[**export_tag_to_label_box_data_rows**](TagsApi.md#export_tag_to_label_box_data_rows) | **GET** /v1/datasets/{datasetId}/tags/{tagId}/export/LabelBoxDataRows | 
[**export_tag_to_label_studio_tasks**](TagsApi.md#export_tag_to_label_studio_tasks) | **GET** /v1/datasets/{datasetId}/tags/{tagId}/export/LabelStudioTasks | 
[**get_filenames_by_tag_id**](TagsApi.md#get_filenames_by_tag_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId}/filenames | 
[**get_tag_by_tag_id**](TagsApi.md#get_tag_by_tag_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId} | 
[**get_tags_by_dataset_id**](TagsApi.md#get_tags_by_dataset_id) | **GET** /v1/datasets/{datasetId}/tags | 
[**perform_tag_arithmetics**](TagsApi.md#perform_tag_arithmetics) | **POST** /v1/datasets/{datasetId}/tags/arithmetics | 
[**perform_tag_arithmetics_bitmask**](TagsApi.md#perform_tag_arithmetics_bitmask) | **POST** /v1/datasets/{datasetId}/tags/arithmetics/bitmask | 
[**update_tag_by_tag_id**](TagsApi.md#update_tag_by_tag_id) | **PUT** /v1/datasets/{datasetId}/tags/{tagId} | 
[**upsize_tags_by_dataset_id**](TagsApi.md#upsize_tags_by_dataset_id) | **POST** /v1/datasets/{datasetId}/tags/upsize | 


# **create_initial_tag_by_dataset_id**
> CreateEntityResponse create_initial_tag_by_dataset_id(dataset_id, initial_tag_create_request)



create the intitial tag for a dataset which then locks the dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.initial_tag_create_request import InitialTagCreateRequest
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    initial_tag_create_request = InitialTagCreateRequest(
        name=TagName("initial-tag"),
        creator=TagCreator("UNKNOWN"),
        img_type=ImageType("full"),
    ) # InitialTagCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_initial_tag_by_dataset_id(dataset_id, initial_tag_create_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
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
**201** | Post successful |  -  |
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
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_create_request import TagCreateRequest
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_create_request = TagCreateRequest(
        name=TagName("initial-tag"),
        prev_tag_id=MongoObjectID("50000000abcdef1234566789"),
        query_tag_id=MongoObjectID("50000000abcdef1234566789"),
        preselected_tag_id=MongoObjectID("50000000abcdef1234566789"),
        bit_mask_data=TagBitMaskData("0x80bda23e9"),
        tot_size=1,
        creator=TagCreator("UNKNOWN"),
        changes=TagChangeData(),
    ) # TagCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_tag_by_dataset_id(dataset_id, tag_create_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
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
**201** | Post successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_tag_by_tag_id**
> delete_tag_by_tag_id(dataset_id, tag_id)



delete a specific tag if its a leaf-tag (e.g is not a dependency of another tag)

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag

    # example passing only required values which don't have defaults set
    try:
        api_instance.delete_tag_by_tag_id(dataset_id, tag_id)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->delete_tag_by_tag_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |

### Return type

void (empty response body)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **export_tag_to_label_box_data_rows**
> LabelBoxDataRows export_tag_to_label_box_data_rows(dataset_id, tag_id)



Export samples of a tag as a json for importing into LabelBox as outlined here; https://docs.labelbox.com/docs/images-json ```openapi\\+warning The image URLs are special in that the resource can be accessed by anyone in posession of said URL for the time specified by the expiresIn query param ``` 

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.label_box_data_rows import LabelBoxDataRows
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag
    expires_in = 1 # int | if defined, the URLs provided will only be valid for amount of seconds from time of issuence (optional)
    preview_example = False # bool | if true, will generate a preview example of how the structure will look (optional) if omitted the server will use the default value of False
    access_control = "default" # str | which access control name to be used (optional) if omitted the server will use the default value of "default"

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.export_tag_to_label_box_data_rows(dataset_id, tag_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->export_tag_to_label_box_data_rows: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.export_tag_to_label_box_data_rows(dataset_id, tag_id, expires_in=expires_in, preview_example=preview_example, access_control=access_control)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->export_tag_to_label_box_data_rows: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |
 **expires_in** | **int**| if defined, the URLs provided will only be valid for amount of seconds from time of issuence | [optional]
 **preview_example** | **bool**| if true, will generate a preview example of how the structure will look | [optional] if omitted the server will use the default value of False
 **access_control** | **str**| which access control name to be used | [optional] if omitted the server will use the default value of "default"

### Return type

[**LabelBoxDataRows**](LabelBoxDataRows.md)

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

# **export_tag_to_label_studio_tasks**
> LabelStudioTasks export_tag_to_label_studio_tasks(dataset_id, tag_id)



Export samples of a tag as a json for importing into LabelStudio as outlined here; https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format ```openapi\\+warning The image URLs are special in that the resource can be accessed by anyone in posession of said URL for the time specified by the expiresIn query param ``` 

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.label_studio_tasks import LabelStudioTasks
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag
    expires_in = 1 # int | if defined, the URLs provided will only be valid for amount of seconds from time of issuence (optional)
    preview_example = False # bool | if true, will generate a preview example of how the structure will look (optional) if omitted the server will use the default value of False
    access_control = "default" # str | which access control name to be used (optional) if omitted the server will use the default value of "default"

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.export_tag_to_label_studio_tasks(dataset_id, tag_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->export_tag_to_label_studio_tasks: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.export_tag_to_label_studio_tasks(dataset_id, tag_id, expires_in=expires_in, preview_example=preview_example, access_control=access_control)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->export_tag_to_label_studio_tasks: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |
 **expires_in** | **int**| if defined, the URLs provided will only be valid for amount of seconds from time of issuence | [optional]
 **preview_example** | **bool**| if true, will generate a preview example of how the structure will look | [optional] if omitted the server will use the default value of False
 **access_control** | **str**| which access control name to be used | [optional] if omitted the server will use the default value of "default"

### Return type

[**LabelStudioTasks**](LabelStudioTasks.md)

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

# **get_filenames_by_tag_id**
> TagFilenamesData get_filenames_by_tag_id(dataset_id, tag_id)



Get list of filenames by tag

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_filenames_data import TagFilenamesData
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_filenames_by_tag_id(dataset_id, tag_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
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
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_data import TagData
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_tag_by_tag_id(dataset_id, tag_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
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
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_data import TagData
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_tags_by_dataset_id(dataset_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
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

# **perform_tag_arithmetics**
> TagArithmeticsResponse perform_tag_arithmetics(dataset_id, tag_arithmetics_request)



performs tag arithmetics to compute a new bitmask out of two existing tags and optionally create a tag for it

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.tag_arithmetics_response import TagArithmeticsResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_arithmetics_request import TagArithmeticsRequest
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_arithmetics_request = TagArithmeticsRequest(
        tag_id1=MongoObjectID("50000000abcdef1234566789"),
        tag_id2=MongoObjectID("50000000abcdef1234566789"),
        operation=TagArithmeticsOperation("UNION"),
        new_tag_name=TagName("initial-tag"),
        creator=TagCreator("UNKNOWN"),
    ) # TagArithmeticsRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.perform_tag_arithmetics(dataset_id, tag_arithmetics_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->perform_tag_arithmetics: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_arithmetics_request** | [**TagArithmeticsRequest**](TagArithmeticsRequest.md)|  |

### Return type

[**TagArithmeticsResponse**](TagArithmeticsResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful, created new tag |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **perform_tag_arithmetics_bitmask**
> TagBitMaskResponse perform_tag_arithmetics_bitmask(dataset_id, tag_arithmetics_request)



Performs tag arithmetics to compute a new bitmask out of two existing tags. Does not create a new tag regardless if newTagName is provided

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.tag_bit_mask_response import TagBitMaskResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_arithmetics_request import TagArithmeticsRequest
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_arithmetics_request = TagArithmeticsRequest(
        tag_id1=MongoObjectID("50000000abcdef1234566789"),
        tag_id2=MongoObjectID("50000000abcdef1234566789"),
        operation=TagArithmeticsOperation("UNION"),
        new_tag_name=TagName("initial-tag"),
        creator=TagCreator("UNKNOWN"),
    ) # TagArithmeticsRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.perform_tag_arithmetics_bitmask(dataset_id, tag_arithmetics_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->perform_tag_arithmetics_bitmask: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_arithmetics_request** | [**TagArithmeticsRequest**](TagArithmeticsRequest.md)|  |

### Return type

[**TagBitMaskResponse**](TagBitMaskResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Success |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_tag_by_tag_id**
> update_tag_by_tag_id(dataset_id, tag_id, tag_update_request)



update information about a specific tag

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from swagger_client.model.tag_update_request import TagUpdateRequest
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag
    tag_update_request = TagUpdateRequest(
        name=TagName("initial-tag"),
    ) # TagUpdateRequest | updated data for tag

    # example passing only required values which don't have defaults set
    try:
        api_instance.update_tag_by_tag_id(dataset_id, tag_id, tag_update_request)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->update_tag_by_tag_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |
 **tag_update_request** | [**TagUpdateRequest**](TagUpdateRequest.md)| updated data for tag |

### Return type

void (empty response body)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **upsize_tags_by_dataset_id**
> CreateEntityResponse upsize_tags_by_dataset_id(dataset_id, tag_upsize_request)



Upsize all tags for the dataset to the current size of the dataset. Use this after adding more samples to a dataset with an initial-tag. | Creates a new tag holding all samples which are not yet in the initial-tag. 

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import tags_api
from swagger_client.model.tag_upsize_request import TagUpsizeRequest
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
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
configuration = swagger_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = tags_api.TagsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_upsize_request = TagUpsizeRequest(
        upsize_tag_name=TagName("initial-tag"),
        upsize_tag_creator=TagCreator("UNKNOWN"),
    ) # TagUpsizeRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.upsize_tags_by_dataset_id(dataset_id, tag_upsize_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling TagsApi->upsize_tags_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_upsize_request** | [**TagUpsizeRequest**](TagUpsizeRequest.md)|  |

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
**201** | Post successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

