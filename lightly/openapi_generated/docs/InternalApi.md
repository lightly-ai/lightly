# lightly.openapi_generated.swagger_client.InternalApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**internal_create_cf_bucket_activity**](InternalApi.md#internal_create_cf_bucket_activity) | **POST** /v1/__internal/cloudfunctions/bucket | 

# **internal_create_cf_bucket_activity**
> internal_create_cf_bucket_activity(body)



notify us about activity on a bucket

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: InternalKeyAuth
configuration = lightly.openapi_generated.swagger_client.Configuration()
configuration.api_key['secret'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['secret'] = 'Bearer'

# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.InternalApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
body = lightly.openapi_generated.swagger_client.CreateCFBucketActivityRequest() # CreateCFBucketActivityRequest | 

try:
    api_instance.internal_create_cf_bucket_activity(body)
except ApiException as e:
    print("Exception when calling InternalApi->internal_create_cf_bucket_activity: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**CreateCFBucketActivityRequest**](CreateCFBucketActivityRequest.md)|  | 

### Return type

void (empty response body)

### Authorization

[InternalKeyAuth](../README.md#InternalKeyAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

