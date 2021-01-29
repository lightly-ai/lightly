# lightly.openapi_generated.swagger_client.TagsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_tag_by_dataset_id**](TagsApi.md#create_tag_by_dataset_id) | **POST** /users/datasets/{datasetId}/tags | 
[**get_tag_by_tag_id**](TagsApi.md#get_tag_by_tag_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId} | 
[**get_tags_by_dataset_id**](TagsApi.md#get_tags_by_dataset_id) | **GET** /users/datasets/{datasetId}/tags | 

# **create_tag_by_dataset_id**
> list[TagData] create_tag_by_dataset_id(body, dataset_id)



create new tag for dataset

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
api_instance = lightly.openapi_generated.swagger_client.TagsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
body = lightly.openapi_generated.swagger_client.Body1() # Body1 | 
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset

try:
    api_response = api_instance.create_tag_by_dataset_id(body, dataset_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TagsApi->create_tag_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Body1**](Body1.md)|  | 
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 

### Return type

[**list[TagData]**](TagData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_tag_by_tag_id**
> list[TagData] get_tag_by_tag_id(dataset_id, tag_id)



Get information about a specific tag

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
api_instance = lightly.openapi_generated.swagger_client.TagsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
tag_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the tag

try:
    api_response = api_instance.get_tag_by_tag_id(dataset_id, tag_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TagsApi->get_tag_by_tag_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **tag_id** | [**MongoObjectID**](.md)| ObjectId of the tag | 

### Return type

[**list[TagData]**](TagData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_tags_by_dataset_id**
> list[TagData] get_tags_by_dataset_id(dataset_id)



Get all tags of a dataset

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
api_instance = lightly.openapi_generated.swagger_client.TagsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset

try:
    api_response = api_instance.get_tags_by_dataset_id(dataset_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling TagsApi->get_tags_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 

### Return type

[**list[TagData]**](TagData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

