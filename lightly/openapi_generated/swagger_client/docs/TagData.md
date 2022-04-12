# TagData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**datasetId** | [**MongoObjectID**](MongoObjectID.md) |  | 
**prevTagId** | **str, none_type** | MongoObjectID or null.  Generally: The prevTagId is this tag&#x27;s parent, i.e. it is a superset of this tag. Sampler: The prevTagId is the initial-tag if there was no preselectedTagId, otherwise, it&#x27;s the preselectedTagId.  | 
**queryTagId** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**preselectedTagId** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**name** | [**TagName**](TagName.md) |  | 
**bitMaskData** | [**TagBitMaskData**](TagBitMaskData.md) |  | 
**totSize** | **int** |  | 
**createdAt** | [**Timestamp**](Timestamp.md) |  | 
**lastModifiedAt** | [**Timestamp**](Timestamp.md) |  | [optional] 
**changes** | [**TagChangeData**](TagChangeData.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

