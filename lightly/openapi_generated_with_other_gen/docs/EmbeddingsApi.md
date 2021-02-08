# lightly.openapi_generated_with_other_gen.openapi_client.EmbeddingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_embeddings_by_sample_id**](EmbeddingsApi.md#get_embeddings_by_sample_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/embeddings | 
[**get_embeddings_csv_write_url_by_id**](EmbeddingsApi.md#get_embeddings_csv_write_url_by_id) | **GET** /v1/datasets/{datasetId}/embeddings/writeCSVUrl | 


# **get_embeddings_by_sample_id**
> [EmbeddingData] get_embeddings_by_sample_id(dataset_id, sample_id)



Get all embeddings of a datasets sample

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import embeddings_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from lightly.openapi_generated_with_other_gen.openapi_client.model.embedding_data import EmbeddingData
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
    api_instance = embeddings_api.EmbeddingsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    sample_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the sample
    mode = "full" # str | if we want everything (full) or just the summaries (optional) if omitted the server will use the default value of "full"

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_by_sample_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_by_sample_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **sample_id** | **MongoObjectID**| ObjectId of the sample |
 **mode** | **str**| if we want everything (full) or just the summaries | [optional] if omitted the server will use the default value of "full"

### Return type

[**[EmbeddingData]**](EmbeddingData.md)

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

# **get_embeddings_csv_write_url_by_id**
> WriteCSVUrlData get_embeddings_csv_write_url_by_id(dataset_id)



Get the signed url to upload an CSVembedding to for a specific dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import embeddings_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.write_csv_url_data import WriteCSVUrlData
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
    api_instance = embeddings_api.EmbeddingsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    name = "default" # str | the sampling requests name to create a signed url for (optional) if omitted the server will use the default value of "default"

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_embeddings_csv_write_url_by_id(dataset_id)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_csv_write_url_by_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.get_embeddings_csv_write_url_by_id(dataset_id, name=name)
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling EmbeddingsApi->get_embeddings_csv_write_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **name** | **str**| the sampling requests name to create a signed url for | [optional] if omitted the server will use the default value of "default"

### Return type

[**WriteCSVUrlData**](WriteCSVUrlData.md)

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

