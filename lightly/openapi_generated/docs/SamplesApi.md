# lightly.openapi_generated.swagger_client.SamplesApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_embeddings_by_sample_id**](SamplesApi.md#get_embeddings_by_sample_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/embeddings | Get all embeddings of a datasets sample
[**get_sample_by_id**](SamplesApi.md#get_sample_by_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId} | Get a specific sample of a dataset
[**get_sample_image_read_url_by_id**](SamplesApi.md#get_sample_image_read_url_by_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/readurl | Get the image path of a specific sample of a dataset
[**get_sample_image_write_url_by_id**](SamplesApi.md#get_sample_image_write_url_by_id) | **GET** /users/datasets/{datasetId}/samples/{sampleId}/writeurl | Get the signed url to upload an image to for a specific sample of a dataset
[**get_samples_by_dataset_id**](SamplesApi.md#get_samples_by_dataset_id) | **GET** /users/datasets/{datasetId}/samples | Get all samples of a dataset
[**update_sample_by_id**](SamplesApi.md#update_sample_by_id) | **PUT** /users/datasets/{datasetId}/samples/{sampleId} | update a specific sample of a dataset

# **get_embeddings_by_sample_id**
> InlineResponse2001 get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)

Get all embeddings of a datasets sample

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplesApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample
mode = 'mode_example' # str | if we want everything (full) or just the summaries (optional)

try:
    # Get all embeddings of a datasets sample
    api_response = api_instance.get_embeddings_by_sample_id(dataset_id, sample_id, mode=mode)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplesApi->get_embeddings_by_sample_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **sample_id** | [**MongoObjectID**](.md)| ObjectId of the sample | 
 **mode** | **str**| if we want everything (full) or just the summaries | [optional] 

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_by_id**
> SampleData get_sample_by_id(dataset_id, sample_id)

Get a specific sample of a dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplesApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample

try:
    # Get a specific sample of a dataset
    api_response = api_instance.get_sample_by_id(dataset_id, sample_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplesApi->get_sample_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **sample_id** | [**MongoObjectID**](.md)| ObjectId of the sample | 

### Return type

[**SampleData**](SampleData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_read_url_by_id**
> str get_sample_image_read_url_by_id(dataset_id, sample_id, type=type)

Get the image path of a specific sample of a dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplesApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample
type = 'type_example' # str | if we want to get the full image or just the thumbnail (optional)

try:
    # Get the image path of a specific sample of a dataset
    api_response = api_instance.get_sample_image_read_url_by_id(dataset_id, sample_id, type=type)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplesApi->get_sample_image_read_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **sample_id** | [**MongoObjectID**](.md)| ObjectId of the sample | 
 **type** | **str**| if we want to get the full image or just the thumbnail | [optional] 

### Return type

**str**

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_image_write_url_by_id**
> InlineResponse200 get_sample_image_write_url_by_id(dataset_id, sample_id, file_name)

Get the signed url to upload an image to for a specific sample of a dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplesApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample
file_name = 'file_name_example' # str | the filename to create a signed url for

try:
    # Get the signed url to upload an image to for a specific sample of a dataset
    api_response = api_instance.get_sample_image_write_url_by_id(dataset_id, sample_id, file_name)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplesApi->get_sample_image_write_url_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **sample_id** | [**MongoObjectID**](.md)| ObjectId of the sample | 
 **file_name** | **str**| the filename to create a signed url for | 

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_samples_by_dataset_id**
> list[SampleData] get_samples_by_dataset_id(dataset_id, mode=mode, filename=filename)

Get all samples of a dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplesApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
mode = 'mode_example' # str | if we want everything (full) or just the ObjectIds (optional)
filename = 'filename_example' # str | filter the samples by filename (optional)

try:
    # Get all samples of a dataset
    api_response = api_instance.get_samples_by_dataset_id(dataset_id, mode=mode, filename=filename)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplesApi->get_samples_by_dataset_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **mode** | **str**| if we want everything (full) or just the ObjectIds | [optional] 
 **filename** | **str**| filter the samples by filename | [optional] 

### Return type

[**list[SampleData]**](SampleData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_sample_by_id**
> SampleData update_sample_by_id(body, dataset_id, sample_id)

update a specific sample of a dataset

### Example
```python
from __future__ import print_function
import time
import lightly.openapi_generated.swagger_client
from lightly.openapi_generated.swagger_client.rest import ApiException
from pprint import pprint


# create an instance of the API class
api_instance = lightly.openapi_generated.swagger_client.SamplesApi(lightly.openapi_generated.swagger_client.ApiClient(configuration))
body = lightly.openapi_generated.swagger_client.Body() # Body | the updated sample to set
dataset_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the dataset
sample_id = lightly.openapi_generated.swagger_client.MongoObjectID() # MongoObjectID | ObjectId of the sample

try:
    # update a specific sample of a dataset
    api_response = api_instance.update_sample_by_id(body, dataset_id, sample_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SamplesApi->update_sample_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Body**](Body.md)| the updated sample to set | 
 **dataset_id** | [**MongoObjectID**](.md)| ObjectId of the dataset | 
 **sample_id** | [**MongoObjectID**](.md)| ObjectId of the sample | 

### Return type

[**SampleData**](SampleData.md)

### Authorization

[auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

