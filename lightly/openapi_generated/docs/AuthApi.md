# openapi_client.AuthApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_tokens**](AuthApi.md#get_tokens) | **GET** /users/tokens | Get auth token from the user


# **get_tokens**
> str get_tokens()

Get auth token from the user

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
    api_instance = openapi_client.AuthApi(api_client)
    
    # example, this endpoint has no required or optional parameters
    try:
        # Get auth token from the user
        api_response = api_instance.get_tokens()
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling AuthApi->get_tokens: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

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

