# swagger_client.AnnotationsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_annotation_by_id**](AnnotationsApi.md#get_annotation_by_id) | **GET** /v1/datasets/{datasetId}/annotations/{annotationId} | 
[**get_annotations_by_dataset_id**](AnnotationsApi.md#get_annotations_by_dataset_id) | **GET** /v1/datasets/{datasetId}/annotations | 

# **get_annotation_by_id**
> AnnotationData get_annotation_by_id(dataset_idannotation_id)



Get a Annotation by its ID

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import annotations_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.annotation_data import AnnotationData
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
    api_instance = annotations_api.AnnotationsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'annotationId': MongoObjectID("50000000abcdef1234566789"),
    }
    try:
        api_response = api_instance.get_annotation_by_id(
            path_params=path_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling AnnotationsApi->get_annotation_by_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
annotationId | AnnotationIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### AnnotationIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | ApiResponseFor200 | Get successful
400 | ApiResponseFor400 | Bad Request / malformed
401 | ApiResponseFor401 | Unauthorized to access this resource
403 | ApiResponseFor403 | Access is forbidden
404 | ApiResponseFor404 | The specified resource was not found

#### ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**AnnotationData**](AnnotationData.md) |  | 


#### ApiResponseFor400
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor400ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor400ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


#### ApiResponseFor401
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor401ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor401ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


#### ApiResponseFor403
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor403ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor403ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


#### ApiResponseFor404
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor404ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor404ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 



[**AnnotationData**](AnnotationData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_annotations_by_dataset_id**
> [AnnotationData] get_annotations_by_dataset_id(dataset_id)



Get all annotations of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import annotations_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.annotation_data import AnnotationData
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
    api_instance = annotations_api.AnnotationsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    try:
        api_response = api_instance.get_annotations_by_dataset_id(
            path_params=path_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling AnnotationsApi->get_annotations_by_dataset_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | ApiResponseFor200 | Get successful
400 | ApiResponseFor400 | Bad Request / malformed
401 | ApiResponseFor401 | Unauthorized to access this resource
403 | ApiResponseFor403 | Access is forbidden
404 | ApiResponseFor404 | The specified resource was not found

#### ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor200ResponseBodyApplicationJson

Type | Description | Notes
------------- | ------------- | -------------
**[AnnotationData]** |  | 

#### ApiResponseFor400
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor400ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor400ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


#### ApiResponseFor401
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor401ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor401ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


#### ApiResponseFor403
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor403ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor403ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


#### ApiResponseFor404
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor404ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor404ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 



[**[AnnotationData]**](AnnotationData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

