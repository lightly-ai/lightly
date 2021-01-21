# swagger_client.AnnotationsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_annotation_by_id**](AnnotationsApi.md#get_annotation_by_id) | **GET** /v1/datasets/{datasetId}/annotations/{annotationId} | Get a Annotation by its ID
[**get_annotations_by_dataset_id**](AnnotationsApi.md#get_annotations_by_dataset_id) | **GET** /v1/datasets/{datasetId}/annotations | Get all annotations of a dataset

# **get_annotation_by_id**
> AnnotationData get_annotation_by_id(dataset_id, annotation_id)

Get a Annotation by its ID

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = swagger_client.AnnotationsApi(swagger_client.ApiClient(configuration))
dataset_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
annotation_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the Annotation

try:
    # Get a Annotation by its ID
    api_response = api_instance.get_annotation_by_id(dataset_id, annotation_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AnnotationsApi->get_annotation_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **annotation_id** | [**MongoObjectID**](.md)| ObjectId of the Annotation | 

### Return type

[**AnnotationData**](AnnotationData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_annotations_by_dataset_id**
> list[AnnotationData] get_annotations_by_dataset_id(dataset_id, annotation_id)

Get all annotations of a dataset

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = swagger_client.AnnotationsApi(swagger_client.ApiClient(configuration))
dataset_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
annotation_id = swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the Annotation

try:
    # Get all annotations of a dataset
    api_response = api_instance.get_annotations_by_dataset_id(dataset_id, annotation_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AnnotationsApi->get_annotations_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **annotation_id** | [**MongoObjectID**](.md)| ObjectId of the Annotation | 

### Return type

[**list[AnnotationData]**](AnnotationData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

