# JobStatusData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**datasetId** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**status** | [**JobState**](JobState.md) |  | 
**meta** | [**JobStatusMeta**](JobStatusMeta.md) |  | [optional] 
**waitTimeTillNextPoll** | **int** | The time in seconds the client should wait before doing the next poll. | 
**createdAt** | [**Timestamp**](Timestamp.md) |  | 
**lastModifiedAt** | [**Timestamp**](Timestamp.md) |  | [optional] 
**finishedAt** | [**Timestamp**](Timestamp.md) |  | [optional] 
**error** | **str** |  | [optional] 
**result** | [**JobStatusDataResult**](JobStatusDataResult.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

