# CropData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**parentId** | [**MongoObjectID**](MongoObjectID.md) |  | 
**predictionIndex** | **int** | the index of this crop within all found prediction singletons of a sampleId (the parentId) | 
**predictionTaskName** | **str** | the name of the prediction task which yielded this crop | 
**predictionTaskCategoryId** | **int** | the categoryId (index) of the categories existing for the prediction task name which yielded this crop | 
**predictionTaskScore** | **int, float** | the score for the prediction task which yielded this crop | 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

