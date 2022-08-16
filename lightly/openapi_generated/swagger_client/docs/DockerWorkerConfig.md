# DockerWorkerConfig

#### Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**workerType** | [**DockerWorkerType**](DockerWorkerType.md) |  | 
**docker** | **{str: (bool, date, datetime, dict, float, int, list, str, none_type)}, none_type** | docker run configurations, keys should match the structure of https://github.com/lightly-ai/lightly-core/blob/develop/onprem-docker/lightly_worker/src/lightly_worker/resources/docker/docker.yaml  | 
**lightly** | **{str: (bool, date, datetime, dict, float, int, list, str, none_type)}, none_type** | lightly configurations which are passed to a docker run, keys should match structure of https://github.com/lightly-ai/lightly/blob/master/lightly/cli/config/config.yaml  | 
**selection** | [**DockerWorkerSelectionConfig**](DockerWorkerSelectionConfig.md) |  | [optional] 
**any string name** | **bool, date, datetime, dict, float, int, list, str, none_type** | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

