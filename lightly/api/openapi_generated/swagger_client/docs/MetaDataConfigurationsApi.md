# swagger_client.MetaDataConfigurationsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_meta_data_configuration**](MetaDataConfigurationsApi.md#create_meta_data_configuration) | **POST** /v1/configuration/metadata | 
[**get_meta_data_configuration_by_id**](MetaDataConfigurationsApi.md#get_meta_data_configuration_by_id) | **GET** /v1/configuration/metadata/{configurationId} | 
[**get_meta_data_configurations**](MetaDataConfigurationsApi.md#get_meta_data_configurations) | **GET** /v1/configuration/metadata | 
[**update_meta_data_configuration_by_id**](MetaDataConfigurationsApi.md#update_meta_data_configuration_by_id) | **PUT** /v1/configuration/metadata/{configurationId} | 


# **create_meta_data_configuration**
> CreateEntityResponse create_meta_data_configuration(configuration_set_request)



Create a new metadata configuration

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import meta_data_configurations_api
from swagger_client.model.create_entity_response import CreateEntityResponse
from swagger_client.model.configuration_set_request import ConfigurationSetRequest
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
    configuration_set_request = ConfigurationSetRequest(
        name="name_example",
        configs=[
            ConfigurationEntry(
                name="name_example",
                path="path_example",
                default_value=None,
                value_data_type=ConfigurationValueDataType("NUMERIC_INT"),
            ),
        ],
    ) # ConfigurationSetRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_meta_data_configuration(configuration_set_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->create_meta_data_configuration: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **configuration_set_request** | [**ConfigurationSetRequest**](ConfigurationSetRequest.md)|  |

### Return type

[**CreateEntityResponse**](CreateEntityResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Post successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_meta_data_configuration_by_id**
> ConfigurationData get_meta_data_configuration_by_id(configuration_id)



Get a specific metadata configuration

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
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
    configuration_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the metadata configuration

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_meta_data_configuration_by_id(configuration_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->get_meta_data_configuration_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **configuration_id** | **MongoObjectID**| ObjectId of the metadata configuration |

### Return type

[**ConfigurationData**](ConfigurationData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_meta_data_configurations**
> [ConfigurationData] get_meta_data_configurations()



Get the all metadata configurations that exist for a user

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import meta_data_configurations_api
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

    # example, this endpoint has no required or optional parameters
    try:
        api_response = api_instance.get_meta_data_configurations()
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->get_meta_data_configurations: %s\n" % e)
```


### Parameters
This endpoint does not need any parameter.

### Return type

[**[ConfigurationData]**](ConfigurationData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_meta_data_configuration_by_id**
> update_meta_data_configuration_by_id(configuration_id, configuration_set_request)



update a specific metadata configuration

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
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
    configuration_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the metadata configuration
    configuration_set_request = ConfigurationSetRequest(
        name="name_example",
        configs=[
            ConfigurationEntry(
                name="name_example",
                path="path_example",
                default_value=None,
                value_data_type=ConfigurationValueDataType("NUMERIC_INT"),
            ),
        ],
    ) # ConfigurationSetRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_instance.update_meta_data_configuration_by_id(configuration_id, configuration_set_request)
    except swagger_client.ApiException as e:
        print("Exception when calling MetaDataConfigurationsApi->update_meta_data_configuration_by_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **configuration_id** | **MongoObjectID**| ObjectId of the metadata configuration |
 **configuration_set_request** | [**ConfigurationSetRequest**](ConfigurationSetRequest.md)|  |

### Return type

void (empty response body)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

