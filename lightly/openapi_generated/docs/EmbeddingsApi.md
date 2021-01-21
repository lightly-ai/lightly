# lightly.openapi_generated.swagger_client.EmbeddingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_embeddings_by_sample_id**](EmbeddingsApi.md#get_embeddings_by_sample_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/embeddings | Get all embeddings of a datasets sample
[**get_embeddings_csv_write_url_by_id**](EmbeddingsApi.md#get_embeddings_csv_write_url_by_id) | **GET** /v1/datasets/{datasetId}/embeddings/writeCSVUrl | Get the signed url to upload an CSVembedding to for a specific dataset

# **get_embeddings_by_sample_id**
> InlineResponse2001 get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)

Get all embeddings of a datasets sample

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.EmbeddingsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample
mode = 'mode_example' # str | if we want everything (full) or just the summaries (optional)

try:
    # Get all embeddings of a datasets sample
    api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EmbeddingsApi->get_embeddings_by_sample_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **sample_id** | [**MongoObjectID**](.md)| ObjectId of the sample | 
 **mode** | **str**| if we want everything (full) or just the summaries | [optional] 

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_embeddings_csv_write_url_by_id**
> InlineResponse2002 get_embeddings_csv_write_url_by_id(dataset_id, name=name)

Get the signed url to upload an CSVembedding to for a specific dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.EmbeddingsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
name = 'name_example' # str | the sampling requests name to create a signed url for (optional)

try:
    # Get the signed url to upload an CSVembedding to for a specific dataset
    api_response = api_instance.get_embeddings_csv_write_url_by_id(dataset_id, name=name)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EmbeddingsApi->get_embeddings_csv_write_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **name** | **str**| the sampling requests name to create a signed url for | [optional] 

### Return type

[**InlineResponse2002**](InlineResponse2002.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

