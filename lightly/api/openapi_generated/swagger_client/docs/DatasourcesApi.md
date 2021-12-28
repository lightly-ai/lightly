# swagger_client.DatasourcesApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_datasource_by_dataset_id**](DatasourcesApi.md#get_datasource_by_dataset_id) | **GET** /v1/datasets/{datasetId}/datasource | 
[**get_datasource_processed_until_timestamp_by_dataset_id**](DatasourcesApi.md#get_datasource_processed_until_timestamp_by_dataset_id) | **GET** /v1/datasets/{datasetId}/datasource/processedUntilTimestamp | 
[**get_list_of_raw_samples_from_datasource_by_dataset_id**](DatasourcesApi.md#get_list_of_raw_samples_from_datasource_by_dataset_id) | **GET** /v1/datasets/{datasetId}/datasource/list | 
[**update_datasource_by_dataset_id**](DatasourcesApi.md#update_datasource_by_dataset_id) | **PUT** /v1/datasets/{datasetId}/datasource | 
[**update_datasource_processed_until_timestamp_by_dataset_id**](DatasourcesApi.md#update_datasource_processed_until_timestamp_by_dataset_id) | **PUT** /v1/datasets/{datasetId}/datasource/processedUntilTimestamp | 
[**verify_datasource_by_dataset_id**](DatasourcesApi.md#verify_datasource_by_dataset_id) | **GET** /v1/datasets/{datasetId}/datasource/verify | 


# **get_datasource_by_dataset_id**
> DatasourceConfig get_datasource_by_dataset_id(dataset_id)



Get the datasource of a dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import datasources_api
from swagger_client.model.datasource_config import DatasourceConfig
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
    api_instance = datasources_api.DatasourcesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_datasource_by_dataset_id(dataset_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->get_datasource_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |

### Return type

[**DatasourceConfig**](DatasourceConfig.md)

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

# **get_datasource_processed_until_timestamp_by_dataset_id**
> DatasourceProcessedUntilTimestampResponse get_datasource_processed_until_timestamp_by_dataset_id(dataset_id)



Get timestamp of last treated resource

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import datasources_api
from swagger_client.model.datasource_processed_until_timestamp_response import DatasourceProcessedUntilTimestampResponse
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
    api_instance = datasources_api.DatasourcesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_datasource_processed_until_timestamp_by_dataset_id(dataset_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->get_datasource_processed_until_timestamp_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |

### Return type

[**DatasourceProcessedUntilTimestampResponse**](DatasourceProcessedUntilTimestampResponse.md)

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

# **get_list_of_raw_samples_from_datasource_by_dataset_id**
> DatasourceRawSamplesData get_list_of_raw_samples_from_datasource_by_dataset_id(dataset_id)



Get list of raw samples from datasource

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import datasources_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.timestamp import Timestamp
from swagger_client.model.datasource_raw_samples_data import DatasourceRawSamplesData
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
    api_instance = datasources_api.DatasourcesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    _from = Timestamp(1577836800) # Timestamp | Unix timestamp, only samples with a creation date after `from` will be returned. This parameter is ignored if `cursor` is specified.  (optional)
    to = Timestamp(1577836800) # Timestamp | Unix timestamp, only samples with a creation date before `to` will be returned. This parameter is ignored if `cursor` is specified.  (optional)
    cursor = "cursor_example" # str | Cursor from previous request, encodes `from` and `to` parameters. Specify to continue reading samples from the list.  (optional)

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_list_of_raw_samples_from_datasource_by_dataset_id(dataset_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->get_list_of_raw_samples_from_datasource_by_dataset_id: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.get_list_of_raw_samples_from_datasource_by_dataset_id(dataset_id, _from=_from, to=to, cursor=cursor)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->get_list_of_raw_samples_from_datasource_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **_from** | **Timestamp**| Unix timestamp, only samples with a creation date after &#x60;from&#x60; will be returned. This parameter is ignored if &#x60;cursor&#x60; is specified.  | [optional]
 **to** | **Timestamp**| Unix timestamp, only samples with a creation date before &#x60;to&#x60; will be returned. This parameter is ignored if &#x60;cursor&#x60; is specified.  | [optional]
 **cursor** | **str**| Cursor from previous request, encodes &#x60;from&#x60; and &#x60;to&#x60; parameters. Specify to continue reading samples from the list.  | [optional]

### Return type

[**DatasourceRawSamplesData**](DatasourceRawSamplesData.md)

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

# **update_datasource_by_dataset_id**
> update_datasource_by_dataset_id(dataset_id, datasource_config)



Update the datasource of a specific dataset

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import datasources_api
from swagger_client.model.datasource_config import DatasourceConfig
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
    api_instance = datasources_api.DatasourcesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    datasource_config = DatasourceConfig(None) # DatasourceConfig | updated datasource configuration for a dataset

    # example passing only required values which don't have defaults set
    try:
        api_instance.update_datasource_by_dataset_id(dataset_id, datasource_config)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->update_datasource_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **datasource_config** | [**DatasourceConfig**](DatasourceConfig.md)| updated datasource configuration for a dataset |

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

# **update_datasource_processed_until_timestamp_by_dataset_id**
> update_datasource_processed_until_timestamp_by_dataset_id(dataset_id, datasource_processed_until_timestamp_request)



Update timestamp of last resource in datapool

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import datasources_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
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
    api_instance = datasources_api.DatasourcesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    datasource_processed_until_timestamp_request = DatasourceProcessedUntilTimestampRequest(
        processed_until_timestamp=Timestamp(1577836800),
    ) # DatasourceProcessedUntilTimestampRequest | The updated timestamp to set

    # example passing only required values which don't have defaults set
    try:
        api_instance.update_datasource_processed_until_timestamp_by_dataset_id(dataset_id, datasource_processed_until_timestamp_request)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->update_datasource_processed_until_timestamp_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **datasource_processed_until_timestamp_request** | [**DatasourceProcessedUntilTimestampRequest**](DatasourceProcessedUntilTimestampRequest.md)| The updated timestamp to set |

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

# **verify_datasource_by_dataset_id**
> DatasourceConfigVerifyData verify_datasource_by_dataset_id(dataset_id)



Test and verify that the configured datasource can be accessed correctly

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import datasources_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.datasource_config_verify_data import DatasourceConfigVerifyData
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
    api_instance = datasources_api.DatasourcesApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.verify_datasource_by_dataset_id(dataset_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling DatasourcesApi->verify_datasource_by_dataset_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |

### Return type

[**DatasourceConfigVerifyData**](DatasourceConfigVerifyData.md)

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

