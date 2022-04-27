# FileNameFormat

When the filename is output, which format shall be used. E.g for a sample called 'frame0.png' that was uploaded from a datasource 's3://my_bucket/datasets/for_lightly/' in the folder 'car/green/' - NAME: car/green/frame0.png - DATASOURCE_FULL: s3://my_bucket/datasets/for_lightly/car/green/frame0.png - REDIRECTED_READ_URL: https://api.lightly.ai/v1/datasets/{datasetId}/samples/{sampleId}/readurlRedirect?publicToken={jsonWebToken}  

Type | Description | Notes
------------- | ------------- | -------------
**str** | When the filename is output, which format shall be used. E.g for a sample called &#x27;frame0.png&#x27; that was uploaded from a datasource &#x27;s3://my_bucket/datasets/for_lightly/&#x27; in the folder &#x27;car/green/&#x27; - NAME: car/green/frame0.png - DATASOURCE_FULL: s3://my_bucket/datasets/for_lightly/car/green/frame0.png - REDIRECTED_READ_URL: https://api.lightly.ai/v1/datasets/{datasetId}/samples/{sampleId}/readurlRedirect?publicToken&#x3D;{jsonWebToken}   | defaults to "NAME",  must be one of ["NAME", "DATASOURCE_FULL", "REDIRECTED_READ_URL", ]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

