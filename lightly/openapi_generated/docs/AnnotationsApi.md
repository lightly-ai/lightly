# openapi_client.AnnotationsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_annotation_by_id**](AnnotationsApi.md#get_annotation_by_id) | **GET** /v1/datasets/{datasetId}/annotations/{annotationId} | Get a Annotation by its ID
[**get_annotations_by_dataset_id**](AnnotationsApi.md#get_annotations_by_dataset_id) | **GET** /v1/datasets/{datasetId}/annotations | Get all annotations of a dataset


# **get_annotation_by_id**
> annotation_data.AnnotationData get_annotation_by_id(dataset_id, annotation_id)

Get a Annotation by its ID

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import openapi_client
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): auth0Bearer
configuration = openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.AnnotationsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    annotation_id = 'annotation_id_example' # str | ObjectId of the Annotation
    
    # example passing only required values which don't have defaults set
    try:
        # Get a Annotation by its ID
        api_response = api_instance.get_annotation_by_id(dataset_id, annotation_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling AnnotationsApi->get_annotation_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **annotation_id** | **str**| ObjectId of the Annotation |

### Return type

[**annotation_data.AnnotationData**](AnnotationData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_annotations_by_dataset_id**
> [annotation_data.AnnotationData] get_annotations_by_dataset_id(dataset_id, annotation_id)

Get all annotations of a dataset

### Example

* Bearer (JWT) Authentication (auth0Bearer):
```python
from __future__ import print_function
import time
import openapi_client
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): auth0Bearer
configuration = openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.AnnotationsApi(api_client)
    dataset_id = 'dataset_id_example' # str | ObjectId of the dataset
    annotation_id = 'annotation_id_example' # str | ObjectId of the Annotation
    
    # example passing only required values which don't have defaults set
    try:
        # Get all annotations of a dataset
        api_response = api_instance.get_annotations_by_dataset_id(dataset_id, annotation_id)
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling AnnotationsApi->get_annotations_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| ObjectId of the dataset |
 **annotation_id** | **str**| ObjectId of the Annotation |

### Return type

[**[annotation_data.AnnotationData]**](AnnotationData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

