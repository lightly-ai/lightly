# lightly.openapi_generated.swagger_client.JobsApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_job_status_by_id**](JobsApi.md#get_job_status_by_id) | **GET** /v1/jobs/{jobId} | 

# **get_job_status_by_id**
> JobStatusData get_job_status_by_id(job_id)



Get status of a specific job

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: ApiKeyAuth
configuration = lightly.openapi_generated.swagger_client.Configuration()
configuration.api_key['token'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['token'] = 'Bearer'

# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.JobsApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
job_id = 'job_id_example' # str | id of the job

try:
    api_response = api_instance.get_job_status_by_id(job_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling JobsApi->get_job_status_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **job_id** | **str**| id of the job | 

### Return type

[**JobStatusData**](JobStatusData.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

