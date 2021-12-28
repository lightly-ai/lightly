# swagger_client.SamplesApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_sample_by_dataset_id**](SamplesApi.md#create_sample_by_dataset_id) | **POST** /v1/datasets/{datasetId}/samples | 
[**get_sample_by_id**](SamplesApi.md#get_sample_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId} | 
[**get_sample_image_read_url_by_id**](SamplesApi.md#get_sample_image_read_url_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/readurl | 
[**get_sample_image_resource_redirect_by_id**](SamplesApi.md#get_sample_image_resource_redirect_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/readurlRedirect | 
[**get_sample_image_write_url_by_id**](SamplesApi.md#get_sample_image_write_url_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/writeurl | 
[**get_sample_image_write_urls_by_id**](SamplesApi.md#get_sample_image_write_urls_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/writeurls | 
[**get_samples_by_dataset_id**](SamplesApi.md#get_samples_by_dataset_id) | **GET** /v1/datasets/{datasetId}/samples | 
[**update_sample_by_id**](SamplesApi.md#update_sample_by_id) | **PUT** /v1/datasets/{datasetId}/samples/{sampleId} | 


# **create_sample_by_dataset_id**
> CreateEntityResponse create_sample_by_dataset_id(dataset_id, sample_create_request)



Create a new sample in a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.sample_create_request import SampleCreateRequest
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_create_request = SampleCreateRequest(
        file_name="file_name_example",
        thumb_name="thumb_name_example",
        exif={},
        meta_data=SampleMetaData(
            custom={},
            dynamic={},
            sharpness=3.14,
            size_in_bytes=1,
            snr=3.14,
            mean=[
                3.14,
            ],
            shape=[
                1,
            ],
            std=[
                3.14,
            ],
            sum_of_squares=[
                3.14,
            ],
            sum_of_values=[
                3.14,
            ],
        ),
        custom_meta_data=CustomSampleMetaData(),
        video_frame_data=VideoFrameData(
            source_video="source_video_example",
            source_video_frame_index=3.14,
            source_video_frame_timestamp=3.14,
        ),
    ) # SampleCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_sample_by_dataset_id(dataset_id, sample_create_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->create_sample_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_create_request** | [**SampleCreateRequest**](SampleCreateRequest.md)|  |

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

# **get_sample_by_id**
> SampleData get_sample_by_id(dataset_id, sample_id)



Get a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.sample_data import SampleData
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_sample_by_id(dataset_id, sample_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |

### Return type

[**SampleData**](SampleData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Post successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_read_url_by_id**
> str get_sample_image_read_url_by_id(dataset_id, sample_id)



Get the image path of a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample
    type = "full" # str | if we want to get the full image or just the thumbnail (optional) if omitted the server will use the default value of "full"

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_sample_image_read_url_by_id(dataset_id, sample_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.get_sample_image_read_url_by_id(dataset_id, sample_id, type=type)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |
 **type** | **str**| if we want to get the full image or just the thumbnail | [optional] if omitted the server will use the default value of "full"

### Return type

**str**

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

# **get_sample_image_resource_redirect_by_id**
> str get_sample_image_resource_redirect_by_id(dataset_id, sample_id, )



This endpoint enables anyone given the correct credentials to access the actual image directly. By creating a readURL for the resource and redirecting to that URL, the client can use this endpoint to always have a way to access the resource as there is no expiration 

### Example

* Api Key Authentication (ApiPublicJWTAuth):

```python
import time
import swagger_client
from swagger_client.api import samples_api
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

# Configure API key authorization: ApiPublicJWTAuth
configuration.api_key['ApiPublicJWTAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiPublicJWTAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_sample_image_resource_redirect_by_id(dataset_id, sample_id, )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_resource_redirect_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |
 **type** | **str**| if we want to get the full image or just the thumbnail | defaults to "full"

### Return type

**str**

### Authorization

[ApiPublicJWTAuth](../README.md#ApiPublicJWTAuth)

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

# **get_sample_image_write_url_by_id**
> str get_sample_image_write_url_by_id(dataset_id, sample_id, )



Get the signed url to upload an image to for a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_sample_image_write_url_by_id(dataset_id, sample_id, )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_write_url_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |
 **is_thumbnail** | **bool**| Whether or not the image to upload is a thumbnail | defaults to False

### Return type

**str**

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

# **get_sample_image_write_urls_by_id**
> SampleWriteUrls get_sample_image_write_urls_by_id(dataset_id, sample_id)



Get all signed write URLs to upload all images (full image and thumbnail) of a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from swagger_client.model.sample_write_urls import SampleWriteUrls
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_sample_image_write_urls_by_id(dataset_id, sample_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_write_urls_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |

### Return type

[**SampleWriteUrls**](SampleWriteUrls.md)

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

# **get_samples_by_dataset_id**
> [SampleData] get_samples_by_dataset_id(dataset_id)



Get all samples of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.sample_data import SampleData
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    mode = "full" # str | if we want everything (full) or just the ObjectIds (optional) if omitted the server will use the default value of "full"
    file_name = "fileName_example" # str | filter the samples by filename (optional)

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_samples_by_dataset_id(dataset_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.get_samples_by_dataset_id(dataset_id, mode=mode, file_name=file_name)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **mode** | **str**| if we want everything (full) or just the ObjectIds | [optional] if omitted the server will use the default value of "full"
 **file_name** | **str**| filter the samples by filename | [optional]

### Return type

[**[SampleData]**](SampleData.md)

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

# **update_sample_by_id**
> update_sample_by_id(dataset_id, sample_id, sample_update_request)



update a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from swagger_client.model.sample_update_request import SampleUpdateRequest
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
    api_instance = samples_api.SamplesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample
    sample_update_request = SampleUpdateRequest(
        file_name="file_name_example",
        thumb_name="thumb_name_example",
        exif={},
        meta_data=SampleMetaData(
            custom={},
            dynamic={},
            sharpness=3.14,
            size_in_bytes=1,
            snr=3.14,
            mean=[
                3.14,
            ],
            shape=[
                1,
            ],
            std=[
                3.14,
            ],
            sum_of_squares=[
                3.14,
            ],
            sum_of_values=[
                3.14,
            ],
        ),
        custom_meta_data=CustomSampleMetaData(),
    ) # SampleUpdateRequest | The updated sample to set
    enable_dataset_update = False # bool |  (optional) if omitted the server will use the default value of False

    # example passing only required values which don't have defaults set
    try:
        api_instance.update_sample_by_id(dataset_id, sample_id, sample_update_request)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->update_sample_by_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_instance.update_sample_by_id(dataset_id, sample_id, sample_update_request, enable_dataset_update=enable_dataset_update)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->update_sample_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |
 **sample_update_request** | [**SampleUpdateRequest**](SampleUpdateRequest.md)| The updated sample to set |
 **enable_dataset_update** | **bool**|  | [optional] if omitted the server will use the default value of False

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

