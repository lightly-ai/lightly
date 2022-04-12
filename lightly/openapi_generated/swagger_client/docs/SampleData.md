# SampleData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**type** | [**SampleType**](SampleType.md) |  | 
**datasetId** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**fileName** | **str** |  | 
**thumbName** | **str, none_type** |  | [optional] 
**exif** | **{str: (bool, date, datetime, dict, float, int, list, str, none_type)}** |  | [optional] 
**index** | **int** |  | [optional] 
**createdAt** | [**Timestamp**](Timestamp.md) |  | [optional] 
**lastModifiedAt** | [**Timestamp**](Timestamp.md) |  | [optional] 
**metaData** | [**SampleMetaData**](SampleMetaData.md) |  | [optional] 
**customMetaData** | [**CustomSampleMetaData**](CustomSampleMetaData.md) |  | [optional] 
**videoFrameData** | [**VideoFrameData**](VideoFrameData.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

