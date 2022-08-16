# swagger_client.SamplesApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_sample_by_dataset_id**](SamplesApi.md#create_sample_by_dataset_id) | **POST** /v1/datasets/{datasetId}/samples | 
[**get_sample_by_id**](SamplesApi.md#get_sample_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId} | 
[**get_sample_image_read_url_by_id**](SamplesApi.md#get_sample_image_read_url_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/readurl | 
[**get_sample_image_resource_redirect_by_id**](SamplesApi.md#get_sample_image_resource_redirect_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/readurlRedirect | 
[**get_sample_image_write_url_by_id**](SamplesApi.md#get_sample_image_write_url_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/writeurl | 
[**get_sample_image_write_urls_by_id**](SamplesApi.md#get_sample_image_write_urls_by_id) | **GET** /v1/datasets/{datasetId}/samples/{sampleId}/writeurls | 
[**get_samples_by_dataset_id**](SamplesApi.md#get_samples_by_dataset_id) | **GET** /v1/datasets/{datasetId}/samples | 
[**get_samples_partial_by_dataset_id**](SamplesApi.md#get_samples_partial_by_dataset_id) | **GET** /v1/datasets/{datasetId}/samples/partial | 
[**update_sample_by_id**](SamplesApi.md#update_sample_by_id) | **PUT** /v1/datasets/{datasetId}/samples/{sampleId} | 

# **create_sample_by_dataset_id**
> CreateEntityResponse create_sample_by_dataset_id(dataset_idsample_create_request)



Create a new sample in a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.sample_create_request import SampleCreateRequest
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    body = SampleCreateRequest(
        file_name="file_name_example",
        thumb_name="thumb_name_example",
        exif=dict(),
        meta_data=SampleMetaData(
            custom=dict(),
            dynamic=dict(),
            sharpness=3.14,
            size_in_bytes=1,
            snr=3.14,
            mean=[
                3.14
            ],
            shape=[
                1
            ],
            std=[
                3.14
            ],
            sum_of_squares=[
                3.14
            ],
            sum_of_values=[
                3.14
            ],
        ),
        custom_meta_data=CustomSampleMetaData(),
        video_frame_data=VideoFrameData(
            source_video="source_video_example",
            source_video_frame_index=3.14,
            source_video_frame_timestamp=3.14,
        ),
        crop_data=CropData(
            parent_id=MongoObjectID("50000000abcdef1234566789"),
            prediction_uuid_timestamp=Timestamp(1577836800000),
            prediction_index=0,
            prediction_task_name=PathSafeName("gB98J"),
            prediction_task_category_id=CategoryId(42),
            prediction_task_score=Score(0.8),
        ),
    )
    try:
        api_response = api_instance.create_sample_by_dataset_id(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->create_sample_by_dataset_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

#### SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**SampleCreateRequest**](SampleCreateRequest.md) |  | 


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
201 | ApiResponseFor201 | Post successful 
400 | ApiResponseFor400 | Bad Request / malformed 
401 | ApiResponseFor401 | Unauthorized to access this resource 
403 | ApiResponseFor403 | Access is forbidden 
404 | ApiResponseFor404 | The specified resource was not found 

#### ApiResponseFor201
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor201ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor201ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateEntityResponse**](CreateEntityResponse.md) |  | 


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



[**CreateEntityResponse**](CreateEntityResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_by_id**
> SampleData get_sample_by_id(dataset_idsample_id)



Get a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.sample_data import SampleData
from swagger_client.model.mongo_object_id import MongoObjectID
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    try:
        api_response = api_instance.get_sample_by_id(
            path_params=path_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_by_id: %s\n" % e)
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
sampleId | SampleIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### SampleIdSchema
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
[**SampleData**](SampleData.md) |  | 


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



[**SampleData**](SampleData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_read_url_by_id**
> str get_sample_image_read_url_by_id(dataset_idsample_id)



Get the image path of a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
    }
    try:
        api_response = api_instance.get_sample_image_read_url_by_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)

    # example passing only optional values
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'type': "full",
    }
    try:
        api_response = api_instance.get_sample_image_read_url_by_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
type | TypeSchema | | optional


#### TypeSchema

Type | Description | Notes
------------- | ------------- | -------------
**str** |  | defaults to "full",  must be one of ["full", "thumbnail", ]

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
sampleId | SampleIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### SampleIdSchema
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
**str** |  | 

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



**str**

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_resource_redirect_by_id**
> get_sample_image_resource_redirect_by_id(dataset_idsample_id)



This endpoint enables anyone given the correct credentials to access the actual image directly. By creating a readURL for the resource and redirecting to that URL, the client can use this endpoint to always have a way to access the resource as there is no expiration 

### Example

* Api Key Authentication (ApiPublicJWTAuth):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
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

# Configure API key authorization: ApiPublicJWTAuth
configuration.api_key['ApiPublicJWTAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiPublicJWTAuth'] = 'Bearer'
# Enter a context with an instance of the API client
with swagger_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'type': "full",
    }
    try:
        api_response = api_instance.get_sample_image_resource_redirect_by_id(
            path_params=path_params,
            query_params=query_params,
        )
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_resource_redirect_by_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
type | TypeSchema | | 


#### TypeSchema

Type | Description | Notes
------------- | ------------- | -------------
**str** |  | defaults to "full",  must be one of ["full", "thumbnail", ]

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
sampleId | SampleIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### SampleIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
307 | ApiResponseFor307 | Temporary redirect 
400 | ApiResponseFor400 | Bad Request / malformed 
401 | ApiResponseFor401 | Unauthorized to access this resource 
403 | ApiResponseFor403 | Access is forbidden 
404 | ApiResponseFor404 | The specified resource was not found 

#### ApiResponseFor307
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor307ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

#### SchemaFor307ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ApiErrorResponse**](ApiErrorResponse.md) |  | 


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



void (empty response body)

### Authorization

[ApiPublicJWTAuth](../README.md#ApiPublicJWTAuth)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_write_url_by_id**
> str get_sample_image_write_url_by_id(dataset_idsample_id)



Get the signed url to upload an image to for a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'isThumbnail': False,
    }
    try:
        api_response = api_instance.get_sample_image_write_url_by_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_write_url_by_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
isThumbnail | IsThumbnailSchema | | 


#### IsThumbnailSchema

Type | Description | Notes
------------- | ------------- | -------------
**bool** |  | defaults to False

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
sampleId | SampleIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### SampleIdSchema
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
**str** |  | 

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



**str**

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_write_urls_by_id**
> SampleWriteUrls get_sample_image_write_urls_by_id(dataset_idsample_id)



Get all signed write URLs to upload all images (full image and thumbnail) of a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from swagger_client.model.sample_write_urls import SampleWriteUrls
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    try:
        api_response = api_instance.get_sample_image_write_urls_by_id(
            path_params=path_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_sample_image_write_urls_by_id: %s\n" % e)
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
sampleId | SampleIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### SampleIdSchema
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
[**SampleWriteUrls**](SampleWriteUrls.md) |  | 


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



[**SampleWriteUrls**](SampleWriteUrls.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_samples_by_dataset_id**
> [SampleData] get_samples_by_dataset_id(dataset_id)



Get all samples of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.sample_data import SampleData
from swagger_client.model.mongo_object_id import MongoObjectID
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
    }
    try:
        api_response = api_instance.get_samples_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)

    # example passing only optional values
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'fileName': "fileName_example",
        'pageSize': 3.14,
        'pageOffset': 3.14,
    }
    try:
        api_response = api_instance.get_samples_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
fileName | FileNameSchema | | optional
pageSize | PageSizeSchema | | optional
pageOffset | PageOffsetSchema | | optional


#### FileNameSchema

Type | Description | Notes
------------- | ------------- | -------------
**str** |  | 

#### PageSizeSchema

Type | Description | Notes
------------- | ------------- | -------------
**int, float** |  | 

#### PageOffsetSchema

Type | Description | Notes
------------- | ------------- | -------------
**int, float** |  | 

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
**[SampleData]** |  | 

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



[**[SampleData]**](SampleData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_samples_partial_by_dataset_id**
> [SampleDataModes] get_samples_partial_by_dataset_id(dataset_id)



Get partial information of all samples of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.sample_partial_mode import SamplePartialMode
from swagger_client.model.sample_data_modes import SampleDataModes
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
    }
    try:
        api_response = api_instance.get_samples_partial_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_partial_by_dataset_id: %s\n" % e)

    # example passing only optional values
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'mode': SamplePartialMode("full"),
        'fileName': "fileName_example",
        'pageSize': 3.14,
        'pageOffset': 3.14,
    }
    try:
        api_response = api_instance.get_samples_partial_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->get_samples_partial_by_dataset_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
mode | ModeSchema | | optional
fileName | FileNameSchema | | optional
pageSize | PageSizeSchema | | optional
pageOffset | PageOffsetSchema | | optional


#### ModeSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**SamplePartialMode**](SamplePartialMode.md) |  | 


#### FileNameSchema

Type | Description | Notes
------------- | ------------- | -------------
**str** |  | 

#### PageSizeSchema

Type | Description | Notes
------------- | ------------- | -------------
**int, float** |  | 

#### PageOffsetSchema

Type | Description | Notes
------------- | ------------- | -------------
**int, float** |  | 

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
**[SampleDataModes]** |  | 

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



[**[SampleDataModes]**](SampleDataModes.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_sample_by_id**
> update_sample_by_id(dataset_idsample_idsample_update_request)



update a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import samples_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.api_error_response import ApiErrorResponse
from swagger_client.model.sample_update_request import SampleUpdateRequest
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
    api_instance = samples_api.SamplesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
    }
    body = SampleUpdateRequest(
        file_name="file_name_example",
        thumb_name="thumb_name_example",
        exif=dict(),
        meta_data=SampleMetaData(
            custom=dict(),
            dynamic=dict(),
            sharpness=3.14,
            size_in_bytes=1,
            snr=3.14,
            mean=[
                3.14
            ],
            shape=[
                1
            ],
            std=[
                3.14
            ],
            sum_of_squares=[
                3.14
            ],
            sum_of_values=[
                3.14
            ],
        ),
        custom_meta_data=CustomSampleMetaData(),
    )
    try:
        api_response = api_instance.update_sample_by_id(
            path_params=path_params,
            query_params=query_params,
            body=body,
        )
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->update_sample_by_id: %s\n" % e)

    # example passing only optional values
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'enableDatasetUpdate': False,
    }
    body = SampleUpdateRequest(
        file_name="file_name_example",
        thumb_name="thumb_name_example",
        exif=dict(),
        meta_data=SampleMetaData(
            custom=dict(),
            dynamic=dict(),
            sharpness=3.14,
            size_in_bytes=1,
            snr=3.14,
            mean=[
                3.14
            ],
            shape=[
                1
            ],
            std=[
                3.14
            ],
            sum_of_squares=[
                3.14
            ],
            sum_of_values=[
                3.14
            ],
        ),
        custom_meta_data=CustomSampleMetaData(),
    )
    try:
        api_response = api_instance.update_sample_by_id(
            path_params=path_params,
            query_params=query_params,
            body=body,
        )
    except swagger_client.ApiException as e:
        print("Exception when calling SamplesApi->update_sample_by_id: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

#### SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**SampleUpdateRequest**](SampleUpdateRequest.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
enableDatasetUpdate | EnableDatasetUpdateSchema | | optional


#### EnableDatasetUpdateSchema

Type | Description | Notes
------------- | ------------- | -------------
**bool** |  | defaults to False

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
sampleId | SampleIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### SampleIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | ApiResponseFor200 | OK 
400 | ApiResponseFor400 | Bad Request / malformed 
401 | ApiResponseFor401 | Unauthorized to access this resource 
403 | ApiResponseFor403 | Access is forbidden 
404 | ApiResponseFor404 | The specified resource was not found 

#### ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | Unset | body was not defined |
headers | Unset | headers were not defined |

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



void (empty response body)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

