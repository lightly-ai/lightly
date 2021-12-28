# SampleDataBase


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**type** | **str** | Type of the sample (Image vs VideoFrame). Determined by the API! | 
**file_name** | **str** |  | 
**dataset_id** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**thumb_name** | **str, none_type** |  | [optional] 
**exif** | **{str: (bool, date, datetime, dict, float, int, list, str, none_type)}** |  | [optional] 
**index** | **int** |  | [optional] 
**created_at** | [**Timestamp**](Timestamp.md) |  | [optional] 
**last_modified_at** | [**Timestamp**](Timestamp.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


