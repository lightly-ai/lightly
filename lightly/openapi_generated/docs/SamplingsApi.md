# swagger_client.SamplingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**trigger_sampling_by_id**](SamplingsApi.md#trigger_sampling_by_id) | **POST** /v1/datasets/{datasetId}/tags/{tagId}/embeddings/{embeddingId}/sampling | Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

# **trigger_sampling_by_id**
> InlineResponse2003 trigger_sampling_by_id(body, dataset_id, tag_id, embedding_id)

Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = swagger_client.SamplingsApi(swagger_client.ApiClient(configuration))
body = swagger_client.SamplingCreateRequest() # SamplingCreateRequest | 
dataset_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
tag_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the tag
embedding_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the prev uploaded embedding

try:
    # Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding
    api_response = api_instance.trigger_sampling_by_id(body, dataset_id, tag_id, embedding_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplingsApi->trigger_sampling_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**SamplingCreateRequest**](SamplingCreateRequest.md)|  | 
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **tag_id** | [**MongoObjectID**](.md)| ObjectId of the tag | 
 **embedding_id** | [**MongoObjectID**](.md)| ObjectId of the prev uploaded embedding | 

### Return type

[**InlineResponse2003**](InlineResponse2003.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

