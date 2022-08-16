# PredictionTaskSchema

The schema for predictions or labels when doing classification, object detection, keypoint detection or instance segmentation 

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | [**PathSafeName**](PathSafeName.md) |  | 
**type** | [**TaskType**](TaskType.md) |  | 
**categories** | **[PredictionTaskSchemaCategory]** | An array of the categories that exist for this prediction task. The id needs to be unique | 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

