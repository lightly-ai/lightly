# lightly.openapi_generated_with_other_gen.openapi_client.InternalApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**internal_create_cf_bucket_activity**](InternalApi.md#internal_create_cf_bucket_activity) | **POST** /v1/__internal/cloudfunctions/bucket | 


# **internal_create_cf_bucket_activity**
> internal_create_cf_bucket_activity(create_cf_bucket_activity_request)



notify us about activity on a bucket

### Example

* Api Key Authentication (InternalKeyAuth):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import internal_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.create_cf_bucket_activity_request import CreateCFBucketActivityRequest
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from pprint import pprint
# Defining the host is optional and defaults to https://api.lightly.ai
# See configuration.py for a list of all supported configuration parameters.
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    host = "https://api.lightly.ai"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: InternalKeyAuth
configuration.api_key['InternalKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['InternalKeyAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = internal_api.InternalApi(api_client)
    create_cf_bucket_activity_request = CreateCFBucketActivityRequest(
        name="name_example",
        bucket="bucket_example",
    ) # CreateCFBucketActivityRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_instance.internal_create_cf_bucket_activity(create_cf_bucket_activity_request)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling InternalApi->internal_create_cf_bucket_activity: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_cf_bucket_activity_request** | [**CreateCFBucketActivityRequest**](CreateCFBucketActivityRequest.md)|  |

### Return type

void (empty response body)

### Authorization

[InternalKeyAuth](../README.md#InternalKeyAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Operation Successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

