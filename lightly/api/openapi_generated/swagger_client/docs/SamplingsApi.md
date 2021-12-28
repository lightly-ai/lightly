# swagger_client.SamplingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**trigger_sampling_by_id**](SamplingsApi.md#trigger_sampling_by_id) | **POST** /v1/datasets/{datasetId}/embeddings/{embeddingId}/sampling | 


# **trigger_sampling_by_id**
> AsyncTaskData trigger_sampling_by_id(dataset_id, embedding_id, sampling_create_request)



Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import samplings_api
from swagger_client.model.async_task_data import AsyncTaskData
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from swagger_client.model.sampling_create_request import SamplingCreateRequest
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
    api_instance = samplings_api.SamplingsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    embedding_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the embedding
    sampling_create_request = SamplingCreateRequest(
        new_tag_name=TagName("initial-tag"),
        method=SamplingMethod("ACTIVE_LEARNING"),
        config=SamplingConfig(
            stopping_condition=SamplingConfigStoppingCondition(
                n_samples=3.14,
                min_distance=3.14,
            ),
        ),
        preselected_tag_id=MongoObjectID("50000000abcdef1234566789"),
        query_tag_id=MongoObjectID("50000000abcdef1234566789"),
        row_count=3.14,
    ) # SamplingCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.trigger_sampling_by_id(dataset_id, embedding_id, sampling_create_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplingsApi->trigger_sampling_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **embedding_id** | **MongoObjectID**| ObjectId of the embedding |
 **sampling_create_request** | [**SamplingCreateRequest**](SamplingCreateRequest.md)|  |

### Return type

[**AsyncTaskData**](AsyncTaskData.md)

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

