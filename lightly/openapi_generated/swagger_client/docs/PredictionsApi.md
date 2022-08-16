# swagger_client.PredictionsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_or_update_prediction_by_sample_id**](PredictionsApi.md#create_or_update_prediction_by_sample_id) | **POST** /v1/datasets/{datasetId}/predictions/samples/{sampleId} | 
[**create_or_update_prediction_task_schema_by_dataset_id**](PredictionsApi.md#create_or_update_prediction_task_schema_by_dataset_id) | **POST** /v1/datasets/{datasetId}/predictions/tasks | 
[**get_prediction_by_sample_id**](PredictionsApi.md#get_prediction_by_sample_id) | **GET** /v1/datasets/{datasetId}/predictions/samples/{sampleId} | 
[**get_prediction_task_schema_by_task_name**](PredictionsApi.md#get_prediction_task_schema_by_task_name) | **GET** /v1/datasets/{datasetId}/predictions/tasks/{taskName} | 
[**get_prediction_task_schemas_by_dataset_id**](PredictionsApi.md#get_prediction_task_schemas_by_dataset_id) | **GET** /v1/datasets/{datasetId}/predictions/tasks | 
[**get_predictions_by_dataset_id**](PredictionsApi.md#get_predictions_by_dataset_id) | **GET** /v1/datasets/{datasetId}/predictions/samples | 

# **create_or_update_prediction_by_sample_id**
> CreateEntityResponse create_or_update_prediction_by_sample_id(dataset_idsample_idprediction_uuid_timestampprediction_singletons)



Create/Update all the prediction singletons for a sampleId in the order/index of them being discovered 

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import predictions_api
from swagger_client.model.prediction_singletons import PredictionSingletons
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.timestamp import Timestamp
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
    api_instance = predictions_api.PredictionsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
    }
    body = PredictionSingletons([
        PredictionSingleton()
    ])
    try:
        api_response = api_instance.create_or_update_prediction_by_sample_id(
            path_params=path_params,
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->create_or_update_prediction_by_sample_id: %s\n" % e)
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
[**PredictionSingletons**](PredictionSingletons.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
predictionUUIDTimestamp | PredictionUUIDTimestampSchema | | 


#### PredictionUUIDTimestampSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**Timestamp**](Timestamp.md) |  | 


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

# **create_or_update_prediction_task_schema_by_dataset_id**
> CreateEntityResponse create_or_update_prediction_task_schema_by_dataset_id(dataset_idprediction_uuid_timestampprediction_task_schema)



Creates/updates a prediction task schema with the task name

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import predictions_api
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.prediction_task_schema import PredictionTaskSchema
from swagger_client.model.timestamp import Timestamp
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
    api_instance = predictions_api.PredictionsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
    }
    body = PredictionTaskSchema(
        name=PathSafeName("gB98J"),
        type=TaskType("CLASSIFICATION"),
        categories=[
            PredictionTaskSchemaCategory(
                id=CategoryId(42),
                name=CategoryName("airplane: Boeing_747"),
            )
        ],
    )
    try:
        api_response = api_instance.create_or_update_prediction_task_schema_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->create_or_update_prediction_task_schema_by_dataset_id: %s\n" % e)
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
[**PredictionTaskSchema**](PredictionTaskSchema.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
predictionUUIDTimestamp | PredictionUUIDTimestampSchema | | 


#### PredictionUUIDTimestampSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**Timestamp**](Timestamp.md) |  | 


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
201 | ApiResponseFor201 | Creation successful 
400 | ApiResponseFor400 | Bad Request / malformed 
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

# **get_prediction_by_sample_id**
> PredictionSingletons get_prediction_by_sample_id(dataset_idsample_idprediction_uuid_timestamp)



Get all prediction singletons of a specific sample of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import predictions_api
from swagger_client.model.prediction_singletons import PredictionSingletons
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.timestamp import Timestamp
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
    api_instance = predictions_api.PredictionsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'sampleId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
    }
    try:
        api_response = api_instance.get_prediction_by_sample_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->get_prediction_by_sample_id: %s\n" % e)
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
predictionUUIDTimestamp | PredictionUUIDTimestampSchema | | 


#### PredictionUUIDTimestampSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**Timestamp**](Timestamp.md) |  | 


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
[**PredictionSingletons**](PredictionSingletons.md) |  | 


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



[**PredictionSingletons**](PredictionSingletons.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_prediction_task_schema_by_task_name**
> PredictionTaskSchema get_prediction_task_schema_by_task_name(dataset_idprediction_uuid_timestamptask_name)



Get a prediction task schemas named taskName for a datasetId

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import predictions_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.prediction_task_schema import PredictionTaskSchema
from swagger_client.model.timestamp import Timestamp
from swagger_client.model.path_safe_name import PathSafeName
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
    api_instance = predictions_api.PredictionsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'taskName': PathSafeName("taskName_example"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
    }
    try:
        api_response = api_instance.get_prediction_task_schema_by_task_name(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->get_prediction_task_schema_by_task_name: %s\n" % e)
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
predictionUUIDTimestamp | PredictionUUIDTimestampSchema | | 


#### PredictionUUIDTimestampSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**Timestamp**](Timestamp.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
taskName | TaskNameSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### TaskNameSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**PathSafeName**](PathSafeName.md) |  | 


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
[**PredictionTaskSchema**](PredictionTaskSchema.md) |  | 


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



[**PredictionTaskSchema**](PredictionTaskSchema.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_prediction_task_schemas_by_dataset_id**
> [PredictionTaskSchema] get_prediction_task_schemas_by_dataset_id(dataset_id)



Get list of all the prediction task schemas for a datasetId at a specific predictionUUIDTimestamp. If no predictionUUIDTimestamp is set, it defaults to the newest

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import predictions_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.prediction_task_schema import PredictionTaskSchema
from swagger_client.model.timestamp import Timestamp
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
    api_instance = predictions_api.PredictionsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
    }
    try:
        api_response = api_instance.get_prediction_task_schemas_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->get_prediction_task_schemas_by_dataset_id: %s\n" % e)

    # example passing only optional values
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
    }
    try:
        api_response = api_instance.get_prediction_task_schemas_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->get_prediction_task_schemas_by_dataset_id: %s\n" % e)
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
predictionUUIDTimestamp | PredictionUUIDTimestampSchema | | optional


#### PredictionUUIDTimestampSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**Timestamp**](Timestamp.md) |  | 


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
**[PredictionTaskSchema]** |  | 

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



[**[PredictionTaskSchema]**](PredictionTaskSchema.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_predictions_by_dataset_id**
> [PredictionSingletons] get_predictions_by_dataset_id(dataset_idprediction_uuid_timestamp)



Get all prediction singletons of all samples of a dataset ordered by the sample mapping

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import predictions_api
from swagger_client.model.prediction_singletons import PredictionSingletons
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.timestamp import Timestamp
from swagger_client.model.path_safe_name import PathSafeName
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
    api_instance = predictions_api.PredictionsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
    }
    try:
        api_response = api_instance.get_predictions_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->get_predictions_by_dataset_id: %s\n" % e)

    # example passing only optional values
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    query_params = {
        'predictionUUIDTimestamp': Timestamp(1577836800000),
        'taskName': PathSafeName("taskName_example"),
    }
    try:
        api_response = api_instance.get_predictions_by_dataset_id(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling PredictionsApi->get_predictions_by_dataset_id: %s\n" % e)
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
predictionUUIDTimestamp | PredictionUUIDTimestampSchema | | 
taskName | TaskNameSchema | | optional


#### PredictionUUIDTimestampSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**Timestamp**](Timestamp.md) |  | 


#### TaskNameSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**PathSafeName**](PathSafeName.md) |  | 


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
**[PredictionSingletons]** |  | 

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



[**[PredictionSingletons]**](PredictionSingletons.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

