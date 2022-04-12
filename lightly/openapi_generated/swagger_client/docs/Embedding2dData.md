# Embedding2dData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**datasetId** | [**MongoObjectID**](MongoObjectID.md) |  | 
**embeddingId** | [**MongoObjectID**](MongoObjectID.md) |  | 
**name** | **str** | Name of the 2d embedding (default is embedding name + __2d) | 
**createdAt** | [**Timestamp**](Timestamp.md) |  | 
**dimensionalityReductionMethod** | [**DimensionalityReductionMethod**](DimensionalityReductionMethod.md) |  | 
**coordinatesDimension1** | [**Embedding2dCoordinates**](Embedding2dCoordinates.md) |  | [optional] 
**coordinatesDimension2** | [**Embedding2dCoordinates**](Embedding2dCoordinates.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

