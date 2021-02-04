# lightly.openapi_generated.swagger_client.SamplingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**trigger_sampling_by_id**](SamplingsApi.md#trigger_sampling_by_id) | **POST** /v1/datasets/{datasetId}/embeddings/{embeddingId}/sampling | 

# **trigger_sampling_by_id**
> AsyncTaskData trigger_sampling_by_id(body, dataset_id, embedding_id)



Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: ApiKeyAuth
configuration = lightly.openapi_generated.swagger_client.Configuration()
configuration.api_key['token'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['token'] = 'Bearer'

# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplingsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
body = lightly.openapi_generated.swagger_client.SamplingCreateRequest() # SamplingCreateRequest | 
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
embedding_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the embedding

try:
    api_response = api_instance.trigger_sampling_by_id(body, dataset_id, embedding_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplingsApi->trigger_sampling_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**SamplingCreateRequest**](SamplingCreateRequest.md)|  | 
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **embedding_id** | [**MongoObjectID**](.md)| ObjectId of the embedding | 

### Return type

[**AsyncTaskData**](AsyncTaskData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

