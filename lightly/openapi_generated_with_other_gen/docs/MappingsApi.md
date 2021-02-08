# lightly.openapi_generated_with_other_gen.openapi_client.MappingsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_sample_mappings_by_dataset_id**](MappingsApi.md#get_sample_mappings_by_dataset_id) | **GET** /v1/datasets/{datasetId}/mappings | 


# **get_sample_mappings_by_dataset_id**
> [str] get_sample_mappings_by_dataset_id(dataset_id, )



Get all samples of a dataset as a list. List index is the index of the sample2bitmask mapping and the value is the 'field' you wanted (e.g _id, fileName)

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):
```python
import time
import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.api import mappings_api
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
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

# Configure API key authorization: ApiKeyAuth
configuration.api_key['ApiKeyAuth'] = 'YOUR_API_KEY'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['ApiKeyAuth'] = 'Bearer'

# Configure Bearer authorization (JWT): auth0Bearer
configuration = lightly.openapi_generated_with_other_gen.openapi_client.Configuration(
    access_token = 'YOUR_BEARER_TOKEN'
)

# Enter a context with an instance of the API client
with lightly.openapi_generated_with_other_gen.openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mappings_api.MappingsApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_sample_mappings_by_dataset_id(dataset_id, )
        pprint(api_response)
    except lightly.openapi_generated_with_other_gen.openapi_client.ApiException as e:
        print("Exception when calling MappingsApi->get_sample_mappings_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **field** | **str**| the field to return as the value | defaults to "fileName"

### Return type

**[str]**

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

