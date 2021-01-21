# openapi_client.SamplesApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_embeddings_by_sample_id**](SamplesApi.md#get_embeddings_by_sample_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/embeddings | Get all embeddings of a datasets sample
[**get_sample_by_id**](SamplesApi.md#get_sample_by_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId} | Get a specific sample of a dataset
[**get_sample_image_read_url_by_id**](SamplesApi.md#get_sample_image_read_url_by_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/readurl | Get the image path of a specific sample of a dataset
[**get_sample_image_write_url_by_id**](SamplesApi.md#get_sample_image_write_url_by_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/writeurl | Get the signed url to upload an image to for a specific sample of a dataset
[**get_samples_by_dataset_id**](SamplesApi.md#get_samples_by_dataset_id) | **GET** /users/datasets/{datasetId}/samples | Get all samples of a dataset
[**update_sample_by_id**](SamplesApi.md#update_sample_by_id) | **PUT** /users/datasets/{datasetId}/samples/{sampleId} | update a specific sample of a dataset


# **get_embeddings_by_sample_id**
> one_ofobjectarray.OneOfobjectarray get_embeddings_by_sample_id(dataset_id, sample_id)

Get all embeddings of a datasets sample

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
    api_instance = openapi_client.SamplesApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    sample_id = 'sample_id_example' # str | ObjectId of the sample
    mode = 'full' # str | if we want everything (full) or just the summaries (optional) if omitted the server will use the default value of 'full'

    # example passing only required values which don't have defaults set
    try:
        # Get all embeddings of a datasets sample
        api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_embeddings_by_sample_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Get all embeddings of a datasets sample
        api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_embeddings_by_sample_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **sample_id** | **str**| ObjectId of the sample |
 **mode** | **str**| if we want everything (full) or just the summaries | [optional] if omitted the server will use the default value of 'full'

### Return type

[**one_ofobjectarray.OneOfobjectarray**](OneOfobjectarray.md)

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

# **get_sample_by_id**
> sample_data.SampleData get_sample_by_id(dataset_id, sample_id)

Get a specific sample of a dataset

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
    api_instance = openapi_client.SamplesApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    sample_id = 'sample_id_example' # str | ObjectId of the sample
    
    # example passing only required values which don't have defaults set
    try:
        # Get a specific sample of a dataset
        api_response = api_instance.get_sample_by_id(dataset_id, sample_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **sample_id** | **str**| ObjectId of the sample |

### Return type

[**sample_data.SampleData**](SampleData.md)

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

# **get_sample_image_read_url_by_id**
> str get_sample_image_read_url_by_id(dataset_id, sample_id)

Get the image path of a specific sample of a dataset

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
    api_instance = openapi_client.SamplesApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    sample_id = 'sample_id_example' # str | ObjectId of the sample
    type = 'full' # str | if we want to get the full image or just the thumbnail (optional) if omitted the server will use the default value of 'full'

    # example passing only required values which don't have defaults set
    try:
        # Get the image path of a specific sample of a dataset
        api_response = api_instance.get_sample_image_read_url_by_id(dataset_id, sample_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Get the image path of a specific sample of a dataset
        api_response = api_instance.get_sample_image_read_url_by_id(dataset_id, sample_id, type=type)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **sample_id** | **str**| ObjectId of the sample |
 **type** | **str**| if we want to get the full image or just the thumbnail | [optional] if omitted the server will use the default value of 'full'

### Return type

**str**

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

# **get_sample_image_write_url_by_id**
> inline_response200.InlineResponse200 get_sample_image_write_url_by_id(dataset_id, sample_id, file_name)

Get the signed url to upload an image to for a specific sample of a dataset

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
    api_instance = openapi_client.SamplesApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    sample_id = 'sample_id_example' # str | ObjectId of the sample
    file_name = 'file_name_example' # str | the filename to create a signed url for
    
    # example passing only required values which don't have defaults set
    try:
        # Get the signed url to upload an image to for a specific sample of a dataset
        api_response = api_instance.get_sample_image_write_url_by_id(dataset_id, sample_id, file_name)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_write_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **sample_id** | **str**| ObjectId of the sample |
 **file_name** | **str**| the filename to create a signed url for |

### Return type

[**inline_response200.InlineResponse200**](InlineResponse200.md)

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

# **get_samples_by_dataset_id**
> [sample_data.SampleData] get_samples_by_dataset_id(dataset_id)

Get all samples of a dataset

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
    api_instance = openapi_client.SamplesApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    mode = 'full' # str | if we want everything (full) or just the ObjectIds (optional) if omitted the server will use the default value of 'full'
filename = 'filename_example' # str | filter the samples by filename (optional)

    # example passing only required values which don't have defaults set
    try:
        # Get all samples of a dataset
        api_response = api_instance.get_samples_by_dataset_id(dataset_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Get all samples of a dataset
        api_response = api_instance.get_samples_by_dataset_id(dataset_id, mode=mode, filename=filename)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **mode** | **str**| if we want everything (full) or just the ObjectIds | [optional] if omitted the server will use the default value of 'full'
 **filename** | **str**| filter the samples by filename | [optional]

### Return type

[**[sample_data.SampleData]**](SampleData.md)

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

# **update_sample_by_id**
> sample_data.SampleData update_sample_by_id(dataset_id, sample_id, inline_object_inline_object)

update a specific sample of a dataset

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
    api_instance = openapi_client.SamplesApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    sample_id = 'sample_id_example' # str | ObjectId of the sample
    inline_object_inline_object = openapi_client.InlineObject() # inline_object.InlineObject | 
    
    # example passing only required values which don't have defaults set
    try:
        # update a specific sample of a dataset
        api_response = api_instance.update_sample_by_id(dataset_id, sample_id, inline_object_inline_object)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling SamplesApi->update_sample_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **sample_id** | **str**| ObjectId of the sample |
 **inline_object_inline_object** | [**inline_object.InlineObject**](InlineObject.md)|  |

### Return type

[**sample_data.SampleData**](SampleData.md)

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

