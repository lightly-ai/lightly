# openapi_client.EmbeddingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_embeddings_by_sample_id**](EmbeddingsApi.md#get_embeddings_by_sample_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/embeddings | Get all embeddings of a datasets sample
[**get_embeddings_csv_write_url_by_id**](EmbeddingsApi.md#get_embeddings_csv_write_url_by_id) | **GET** /v1/datasets/{datasetId}/embeddings/writeCSVUrl | Get the signed url to upload an CSVembedding to for a specific dataset


# **get_embeddings_by_sample_id**
> one_ofobjectarray.OneOfobjectarray get_embeddings_by_sample_id(dataset_id, sample_id)

Get all embeddings of a datasets sample

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import openapi_client
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
    api_instance = openapi_client.EmbeddingsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    sample_id = 'sample_id_example' # str | ObjectId of the sample
    mode = 'full' # str | if we want everything (full) or just the summaries (optional) if omitted the server will use the default value of 'full'

    # example passing only required values which don't have defaults set
    try:
        # Get all embeddings of a datasets sample
        api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_by_sample_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Get all embeddings of a datasets sample
        api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_by_sample_id: %s\n" % e)
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

# **get_embeddings_csv_write_url_by_id**
> inline_response2001.InlineResponse2001 get_embeddings_csv_write_url_by_id(dataset_id)

Get the signed url to upload an CSVembedding to for a specific dataset

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import openapi_client
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
    api_instance = openapi_client.EmbeddingsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    name = 'name_example' # str | the sampling requests name to create a signed url for (optional)

    # example passing only required values which don't have defaults set
    try:
        # Get the signed url to upload an CSVembedding to for a specific dataset
        api_response = api_instance.get_embeddings_csv_write_url_by_id(dataset_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_csv_write_url_by_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Get the signed url to upload an CSVembedding to for a specific dataset
        api_response = api_instance.get_embeddings_csv_write_url_by_id(dataset_id, name=name)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_csv_write_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **name** | **str**| the sampling requests name to create a signed url for | [optional]

### Return type

[**inline_response2001.InlineResponse2001**](InlineResponse2001.md)

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

