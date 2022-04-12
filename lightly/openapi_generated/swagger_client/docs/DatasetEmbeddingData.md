# DatasetEmbeddingData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**name** | **str** | name of the embedding chosen by the user calling writeCSVUrl | 
**isProcessed** | **bool** | indicator whether embeddings have already been processed by a background worker | 
**createdAt** | [**Timestamp**](Timestamp.md) |  | 
**is2d** | **bool** | flag set by the background worker if the embedding is 2d | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

