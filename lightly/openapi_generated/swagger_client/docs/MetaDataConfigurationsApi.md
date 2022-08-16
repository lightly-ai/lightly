# swagger_client.MetaDataConfigurationsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_meta_data_configuration**](MetaDataConfigurationsApi.md#create_meta_data_configuration) | **POST** /v1/datasets/{datasetId}/configuration/metadata | 
[**get_meta_data_configuration_by_id**](MetaDataConfigurationsApi.md#get_meta_data_configuration_by_id) | **GET** /v1/datasets/{datasetId}/configuration/metadata/{configurationId} | 
[**get_meta_data_configurations**](MetaDataConfigurationsApi.md#get_meta_data_configurations) | **GET** /v1/datasets/{datasetId}/configuration/metadata | 
[**update_meta_data_configuration_by_id**](MetaDataConfigurationsApi.md#update_meta_data_configuration_by_id) | **PUT** /v1/datasets/{datasetId}/configuration/metadata/{configurationId} | 

# **create_meta_data_configuration**
> CreateEntityResponse create_meta_data_configuration(dataset_idconfiguration_set_request)



Create a new metadata configuration

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import meta_data_configurations_api
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.configuration_set_request import ConfigurationSetRequest
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
    api_instance = meta_data_configurations_api.MetaDataConfigurationsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    body = ConfigurationSetRequest(
        name="name_example",
        configs=[
            ConfigurationEntry(
                name="name_example",
                path="path_example",
                default_value=None,
                value_data_type=ConfigurationValueDataType("NUMERIC_INT"),
            )
        ],
    )
    try:
        api_response = api_instance.create_meta_data_configuration(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->create_meta_data_configuration: %s\n" % e)
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
[**ConfigurationSetRequest**](ConfigurationSetRequest.md) |  | 


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

# **get_meta_data_configuration_by_id**
> ConfigurationData get_meta_data_configuration_by_id(dataset_idconfiguration_id)



Get a specific metadata configuration

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import meta_data_configurations_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.configuration_data import ConfigurationData
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
    api_instance = meta_data_configurations_api.MetaDataConfigurationsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'configurationId': MongoObjectID("50000000abcdef1234566789"),
    }
    try:
        api_response = api_instance.get_meta_data_configuration_by_id(
            path_params=path_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->get_meta_data_configuration_by_id: %s\n" % e)
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
configurationId | ConfigurationIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### ConfigurationIdSchema
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
[**ConfigurationData**](ConfigurationData.md) |  | 


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



[**ConfigurationData**](ConfigurationData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_meta_data_configurations**
> [ConfigurationData] get_meta_data_configurations(dataset_id)



Get the all metadata configurations that exist for a user

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import meta_data_configurations_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.configuration_data import ConfigurationData
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
    api_instance = meta_data_configurations_api.MetaDataConfigurationsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
    }
    try:
        api_response = api_instance.get_meta_data_configurations(
            path_params=path_params,
        )
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->get_meta_data_configurations: %s\n" % e)
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
**[ConfigurationData]** |  | 

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



[**[ConfigurationData]**](ConfigurationData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_meta_data_configuration_by_id**
> update_meta_data_configuration_by_id(dataset_idconfiguration_idconfiguration_set_request)



update a specific metadata configuration

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import swagger_client
from swagger_client.api import meta_data_configurations_api
from swagger_client.model.configuration_set_request import ConfigurationSetRequest
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
    api_instance = meta_data_configurations_api.MetaDataConfigurationsApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'datasetId': MongoObjectID("50000000abcdef1234566789"),
        'configurationId': MongoObjectID("50000000abcdef1234566789"),
    }
    body = ConfigurationSetRequest(
        name="name_example",
        configs=[
            ConfigurationEntry(
                name="name_example",
                path="path_example",
                default_value=None,
                value_data_type=ConfigurationValueDataType("NUMERIC_INT"),
            )
        ],
    )
    try:
        api_response = api_instance.update_meta_data_configuration_by_id(
            path_params=path_params,
            body=body,
        )
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->update_meta_data_configuration_by_id: %s\n" % e)
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
[**ConfigurationSetRequest**](ConfigurationSetRequest.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
datasetId | DatasetIdSchema | | 
configurationId | ConfigurationIdSchema | | 

#### DatasetIdSchema
Type | Description  | Notes
------------- | ------------- | -------------
[**MongoObjectID**](MongoObjectID.md) |  | 


#### ConfigurationIdSchema
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

