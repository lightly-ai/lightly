# DatasetData

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**name** | [**DatasetName**](DatasetName.md) |  | 
**userId** | **str** | The owner of the dataset | 
**accessType** | [**SharedAccessType**](SharedAccessType.md) |  | [optional] 
**type** | [**DatasetType**](DatasetType.md) |  | 
**imgType** | [**ImageType**](ImageType.md) |  | [optional] 
**nSamples** | **int** |  | 
**sizeInBytes** | **int** |  | 
**metaDataConfigurationId** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**createdAt** | [**Timestamp**](Timestamp.md) |  | 
**lastModifiedAt** | [**Timestamp**](Timestamp.md) |  | 
**datasourceProcessedUntilTimestamp** | [**TimestampSeconds**](TimestampSeconds.md) |  | [optional] 
**accessRole** | [**AccessRole**](AccessRole.md) |  | [optional] 
**parentDatasetId** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

