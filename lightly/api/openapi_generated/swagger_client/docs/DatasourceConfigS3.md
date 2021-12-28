# DatasourceConfigS3


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**s3_region** | **str** | the region where your bucket is located (see https://docs.aws.amazon.com/general/latest/gr/s3.html for further information) | 
**s3_access_key_id** | **str** | the accessKeyId of the credential you are providing Lightly to use | 
**s3_secret_access_key** | **str** | the secretAccessKey of the credential you are providing Lightly to use | 
**type** | **str** |  | 
**full_path** | **str** | path includes the bucket name and the path within the bucket where you have stored your information | 
**thumb_suffix** | **str** | the suffix of where to find the thumbnail image. If none is provided, the full image will be loaded where thumbnails would be loaded otherwise. - [filename]: represents the filename without the extension - [extension]: represents the files extension (e.g jpg, png, webp)  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


