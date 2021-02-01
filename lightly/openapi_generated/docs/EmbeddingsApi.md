# swagger_client.EmbeddingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_embeddings_by_sample_id**](EmbeddingsApi.md#get_embeddings_by_sample_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/embeddings | 
[**get_embeddings_csv_write_url_by_id**](EmbeddingsApi.md#get_embeddings_csv_write_url_by_id) | **GET** /v1/datasets/{datasetId}/embeddings/writeCSVUrl | 

# **get_embeddings_by_sample_id**
> list[EmbeddingData] get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)



Get all embeddings of a datasets sample

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: ApiKeyAuth
configuration = swagger_client.Configuration()
configuration.api_key['token'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['token'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.EmbeddingsApi(swagger_client.ApiClient(configuration))
dataset_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample
mode = 'mode_example' # str | if we want everything (full) or just the summaries (optional)

try:
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

[**list[EmbeddingData]**](EmbeddingData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_embeddings_csv_write_url_by_id**
> WriteCSVUrlData get_embeddings_csv_write_url_by_id(dataset_id, name=name)



Get the signed url to upload an CSVembedding to for a specific dataset

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: ApiKeyAuth
configuration = swagger_client.Configuration()
configuration.api_key['token'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['token'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.EmbeddingsApi(swagger_client.ApiClient(configuration))
dataset_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
name = 'default' # str | the sampling requests name to create a signed url for (optional) (default to default)

try:
    api_response = api_instance.get_embeddings_csv_write_url_by_id(dataset_id, name=name)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EmbeddingsApi->get_embeddings_csv_write_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **name** | **str**| the sampling requests name to create a signed url for | [optional] [default to default]

### Return type

[**WriteCSVUrlData**](WriteCSVUrlData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

