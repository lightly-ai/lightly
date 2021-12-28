# swagger_client.ScoresApi

All URIs are relative to *https://api.lightly.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_or_update_active_learning_score_by_tag_id**](ScoresApi.md#create_or_update_active_learning_score_by_tag_id) | **POST** /v1/datasets/{datasetId}/tags/{tagId}/scores | 
[**get_active_learning_score_by_score_id**](ScoresApi.md#get_active_learning_score_by_score_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId}/scores/{scoreId} | 
[**get_active_learning_scores_by_tag_id**](ScoresApi.md#get_active_learning_scores_by_tag_id) | **GET** /v1/datasets/{datasetId}/tags/{tagId}/scores | 


# **create_or_update_active_learning_score_by_tag_id**
> CreateEntityResponse create_or_update_active_learning_score_by_tag_id(dataset_id, tag_id, active_learning_score_create_request)



Create or update active learning score object by tag id

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import scores_api
from swagger_client.model.active_learning_score_create_request import ActiveLearningScoreCreateRequest
from swagger_client.model.create_entity_response import CreateEntityResponse
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
    api_instance = scores_api.ScoresApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag
    active_learning_score_create_request = ActiveLearningScoreCreateRequest(
        score_type=ActiveLearningScoreType("uncertainty_margin"),
        scores=ActiveLearningScores([0.9,0.2,0.5]),
    ) # ActiveLearningScoreCreateRequest | 

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.create_or_update_active_learning_score_by_tag_id(dataset_id, tag_id, active_learning_score_create_request)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling ScoresApi->create_or_update_active_learning_score_by_tag_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |
 **active_learning_score_create_request** | [**ActiveLearningScoreCreateRequest**](ActiveLearningScoreCreateRequest.md)|  |

### Return type

[**CreateEntityResponse**](CreateEntityResponse.md)

### Authorization

[ApiKeyAuth](../README.md#ApiKeyAuth), [auth0Bearer](../README.md#auth0Bearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Post successful |  -  |
**400** | Bad Request / malformed |  -  |
**401** | Unauthorized to access this resource |  -  |
**403** | Access is forbidden |  -  |
**404** | The specified resource was not found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_active_learning_score_by_score_id**
> ActiveLearningScoreData get_active_learning_score_by_score_id(dataset_id, tag_id, score_id)



Get active learning score object by id

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import scores_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.active_learning_score_data import ActiveLearningScoreData
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
    api_instance = scores_api.ScoresApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag
    score_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the scores

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_active_learning_score_by_score_id(dataset_id, tag_id, score_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling ScoresApi->get_active_learning_score_by_score_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |
 **score_id** | **MongoObjectID**| ObjectId of the scores |

### Return type

[**ActiveLearningScoreData**](ActiveLearningScoreData.md)

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

# **get_active_learning_scores_by_tag_id**
> [TagActiveLearningScoresData] get_active_learning_scores_by_tag_id(dataset_id, tag_id)



Get all scoreIds for the given tag

### Example

* Api Key Authentication (ApiKeyAuth):
* Bearer (JWT) Authentication (auth0Bearer):

```python
import time
import swagger_client
from swagger_client.api import scores_api
from swagger_client.model.mongo_object_id import MongoObjectID
from swagger_client.model.tag_active_learning_scores_data import TagActiveLearningScoresData
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
    api_instance = scores_api.ScoresApi(api_client)
    dataset_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the dataset
    tag_id = MongoObjectID("50000000abcdef1234566789") # MongoObjectID | ObjectId of the tag

    # example passing only required values which don't have defaults set
    try:
        api_response = api_instance.get_active_learning_scores_by_tag_id(dataset_id, tag_id)
        pprint(api_response)
    except swagger_client.ApiException as e:
        print("Exception when calling ScoresApi->get_active_learning_scores_by_tag_id: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **MongoObjectID**| ObjectId of the dataset |
 **tag_id** | **MongoObjectID**| ObjectId of the tag |

### Return type

[**[TagActiveLearningScoresData]**](TagActiveLearningScoresData.md)

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

