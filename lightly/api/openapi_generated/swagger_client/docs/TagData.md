# TagData


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**dataset_id** | [**MongoObjectID**](MongoObjectID.md) |  | 
**prev_tag_id** | **str, none_type** | MongoObjectID or null.  Generally: The prevTagId is this tag&#39;s parent, i.e. it is a superset of this tag. Sampler: The prevTagId is the initial-tag if there was no preselectedTagId, otherwise, it&#39;s the preselectedTagId.  | 
**name** | [**TagName**](TagName.md) |  | 
**bit_mask_data** | [**TagBitMaskData**](TagBitMaskData.md) |  | 
**tot_size** | **int** |  | 
**created_at** | [**Timestamp**](Timestamp.md) |  | 
**query_tag_id** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**preselected_tag_id** | [**MongoObjectID**](MongoObjectID.md) |  | [optional] 
**last_modified_at** | [**Timestamp**](Timestamp.md) |  | [optional] 
**changes** | [**TagChangeData**](TagChangeData.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


