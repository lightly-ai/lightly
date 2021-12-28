# JobStatusData


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**status** | [**JobState**](JobState.md) |  | 
**wait_time_till_next_poll** | **int** | The time in seconds the client should wait before doing the next poll. | 
**created_at** | [**Timestamp**](Timestamp.md) |  | 
**meta** | [**JobStatusMeta**](JobStatusMeta.md) |  | [optional] 
**finished_at** | [**Timestamp**](Timestamp.md) |  | [optional] 
**error** | **str** |  | [optional] 
**result** | [**JobStatusDataResult**](JobStatusDataResult.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


