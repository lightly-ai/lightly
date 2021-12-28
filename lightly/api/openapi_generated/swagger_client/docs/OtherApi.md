# swagger_client.OtherApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_ping**](OtherApi.md#get_ping) | **GET** /ping | 


# **get_ping**
> get_ping()



Simple ping to see if the server is available

### Example


```python
import time
import swagger_client
from swagger_client.api import other_api
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
    api_instance = other_api.OtherApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        api_instance.get_ping()
    except swagger_client.ApiException as e:
        print("Exception when calling OtherApi->get_ping: %s\n" % e)
```


### Parameters
This endpoint does not need any parameter.

### Return type

void (empty response body)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request / malformed |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

