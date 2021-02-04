# lightly.openapi_generated.swagger_client.DatasetsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_dataset_by_id**](DatasetsApi.md#get_dataset_by_id) | **GET** /users/datasets/{datasetId} | 
[**get_datasets**](DatasetsApi.md#get_datasets) | **GET** /users/datasets | 

# **get_dataset_by_id**
> DatasetData get_dataset_by_id(dataset_id)



Get a specific dataset

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
api_instance = lightly.openapi_generated.swagger_client.DatasetsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset

try:
    api_response = api_instance.get_dataset_by_id(dataset_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DatasetsApi->get_dataset_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 

### Return type

[**DatasetData**](DatasetData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_datasets**
> list[DatasetData] get_datasets()



Get all datasets for a user

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
api_instance = lightly.openapi_generated.swagger_client.DatasetsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))

try:
    api_response = api_instance.get_datasets()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DatasetsApi->get_datasets: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**list[DatasetData]**](DatasetData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

