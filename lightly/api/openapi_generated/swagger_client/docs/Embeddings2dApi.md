# swagger_client.Embeddings2dApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_embeddings2d_by_embedding_id**](Embeddings2dApi.md#create_embeddings2d_by_embedding_id) | **POST** /v1/datasets/{datasetId}/embeddings/{embeddingId}/2d | 
[**get_embedding2d_by_id**](Embeddings2dApi.md#get_embedding2d_by_id) | **GET** /v1/datasets/{datasetId}/embeddings/{embeddingId}/2d/{embedding2dId} | 
[**get_embeddings2d_by_embedding_id**](Embeddings2dApi.md#get_embeddings2d_by_embedding_id) | **GET** /v1/datasets/{datasetId}/embeddings/{embeddingId}/2d | 


# **create_embeddings2d_by_embedding_id**
> CreateEntityResponse create_embeddings2d_by_embedding_id(dataset_id, embedding_id, embedding2d_create_request)



Create a new 2d embedding

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import embeddings2d_api
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.embedding2d_create_request import Embedding2dCreateRequest
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
    api_instance = embeddings2d_api.Embeddings2dApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    embedding_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the embedding
    embedding2d_create_request = Embedding2dCreateRequest(
        name="name_example",
        dimensionality_reduction_method=DimensionalityReductionMethod("PCA"),
        coordinates_dimension1=Embedding2dCoordinates([0.9,0.2,0.5]),
        coordinates_dimension2=Embedding2dCoordinates([0.9,0.2,0.5]),
    ) # Embedding2dCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_embeddings2d_by_embedding_id(dataset_id, embedding_id, embedding2d_create_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling Embeddings2dApi->create_embeddings2d_by_embedding_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **embedding_id** | **MongoObjectID**| ObjectId of the embedding |
 **embedding2d_create_request** | [**Embedding2dCreateRequest**](Embedding2dCreateRequest.md)|  |

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

# **get_embedding2d_by_id**
> Embedding2dData get_embedding2d_by_id(dataset_id, embedding_id, embedding2d_id)



Get the 2d embeddings by id

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import embeddings2d_api
from swagger_client.model.embedding2d_data import Embedding2dData
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
    api_instance = embeddings2d_api.Embeddings2dApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    embedding_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the embedding
    embedding2d_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the 2d embedding

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_embedding2d_by_id(dataset_id, embedding_id, embedding2d_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling Embeddings2dApi->get_embedding2d_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **embedding_id** | **MongoObjectID**| ObjectId of the embedding |
 **embedding2d_id** | **MongoObjectID**| ObjectId of the 2d embedding |

### Return type

[**Embedding2dData**](Embedding2dData.md)

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

# **get_embeddings2d_by_embedding_id**
> [Embedding2dData] get_embeddings2d_by_embedding_id(dataset_id, embedding_id)



Get all 2d embeddings of an embedding

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import embeddings2d_api
from swagger_client.model.embedding2d_data import Embedding2dData
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
    api_instance = embeddings2d_api.Embeddings2dApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    embedding_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the embedding

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_embeddings2d_by_embedding_id(dataset_id, embedding_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling Embeddings2dApi->get_embeddings2d_by_embedding_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **embedding_id** | **MongoObjectID**| ObjectId of the embedding |

### Return type

[**[Embedding2dData]**](Embedding2dData.md)

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

