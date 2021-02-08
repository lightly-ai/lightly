# JobStatusData

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**status** | [**JobState**](JobState.md) |  | 
**wait_time_till_next_poll** | **int** | The time in seconds the client should wait before doing the next poll. | 
**created_at** | **int** |  | 
**finished_at** | **int** |  | [optional] 
**error** | **str** |  | [optional] 
**result** | [**JobStatusDataResult**](JobStatusDataResult.md) |  | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


