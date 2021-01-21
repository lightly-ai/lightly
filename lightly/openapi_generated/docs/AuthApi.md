# lightly.openapi_generated.swagger_client.AuthApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_tokens**](AuthApi.md#get_tokens) | **GET** /users/tokens | Get auth token from the user

# **get_tokens**
> str get_tokens()

Get auth token from the user

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.AuthApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))

try:
    # Get auth token from the user
    api_response = api_instance.get_tokens()
    pprint(api_response)
except ApiException as e:
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

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

