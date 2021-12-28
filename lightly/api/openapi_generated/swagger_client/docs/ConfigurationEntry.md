# ConfigurationEntry


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | the name of this entry which is displayed in the UI | 
**path** | **str** | the path is the dotnotation which is used to easily access the customMetadata JSON structure of a sample e.g myArray[0].myObject.field | 
**default_value** | **bool, date, datetime, dict, float, int, list, str, none_type** | the default value used if its not possible to extract the value using the path or if the value extracted is nullish | 
**value_data_type** | [**ConfigurationValueDataType**](ConfigurationValueDataType.md) |  | 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


