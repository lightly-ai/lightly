# swagger_client.VersioningApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_latest_pip_version**](VersioningApi.md#get_latest_pip_version) | **GET** /v1/versions/pip/latest | 
[**get_minimum_compatible_pip_version**](VersioningApi.md#get_minimum_compatible_pip_version) | **GET** /v1/versions/pip/minimum | 


# **get_latest_pip_version**
> VersionNumber get_latest_pip_version()



Get latest pip version available

### Example


```python
import time
import swagger_client
from swagger_client.api import versioning_api
from swagger_client.model.version_number import VersionNumber
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
    host = "https://api.lightly.ai"
)


# Enter a context with an instance of the API client
with swagger_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = versioning_api.VersioningApi(api_client)
    current_version = "currentVersion_example" # str |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        api_response = api_instance.get_latest_pip_version(current_version=current_version)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling VersioningApi->get_latest_pip_version: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **current_version** | **str**|  | [optional]

### Return type

[**VersionNumber**](VersionNumber.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_minimum_compatible_pip_version**
> VersionNumber get_minimum_compatible_pip_version()



Get minimum pip version needed for compatability

### Example


```python
import time
import swagger_client
from swagger_client.api import versioning_api
from swagger_client.model.version_number import VersionNumber
from swagger_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = swagger_client.Configuration(
    host = "https://api.lightly.ai"
)


# Enter a context with an instance of the API client
with swagger_client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = versioning_api.VersioningApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        api_response = api_instance.get_minimum_compatible_pip_version()
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling VersioningApi->get_minimum_compatible_pip_version: %s\n" % e)
```


### Parameters
This endpoint does not need any parameter.

### Return type

[**VersionNumber**](VersionNumber.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Get successful |  -  |
**400** | Bad Request / malformed |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

