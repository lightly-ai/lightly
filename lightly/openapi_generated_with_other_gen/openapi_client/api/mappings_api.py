"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


import re  # noqa: F401
import sys  # noqa: F401

from lightly.openapi_generated_with_other_gen.openapi_client.api_client import ApiClient, Endpoint as _Endpoint
from lightly.openapi_generated_with_other_gen.openapi_client.model_utils import (  # noqa: F401
    check_allowed_values,
    check_validations,
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types
)
from lightly.openapi_generated_with_other_gen.openapi_client.model.api_error_response import ApiErrorResponse
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID


class MappingsApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __get_sample_mappings_by_dataset_id(
            self,
            dataset_id,
            field="fileName",
            **kwargs
        ):
            """get_sample_mappings_by_dataset_id  # noqa: E501

            Get all samples of a dataset as a list. List index is the index of the sample2bitmask mapping and the value is the 'field' you wanted (e.g _id, fileName)  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.get_sample_mappings_by_dataset_id(dataset_id, field="fileName", async_req=True)
            >>> result = thread.get()

            Args:
                dataset_id (MongoObjectID): ObjectId of the dataset
                field (str): the field to return as the value. defaults to "fileName", must be one of ["fileName"]

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (float/tuple): timeout setting for this request. If one
                    number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                _host_index (int/None): specifies the index of the server
                    that we want to use.
                    Default is read from the configuration.
                async_req (bool): execute request asynchronously

            Returns:
                [str]
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs['async_req'] = kwargs.get(
                'async_req', False
            )
            kwargs['_return_http_data_only'] = kwargs.get(
                '_return_http_data_only', True
            )
            kwargs['_preload_content'] = kwargs.get(
                '_preload_content', True
            )
            kwargs['_request_timeout'] = kwargs.get(
                '_request_timeout', None
            )
            kwargs['_check_input_type'] = kwargs.get(
                '_check_input_type', True
            )
            kwargs['_check_return_type'] = kwargs.get(
                '_check_return_type', True
            )
            kwargs['_host_index'] = kwargs.get('_host_index')
            kwargs['dataset_id'] = \
                dataset_id
            kwargs['field'] = \
                field
            return self.call_with_http_info(**kwargs)

        self.get_sample_mappings_by_dataset_id = _Endpoint(
            settings={
                'response_type': ([str],),
                'auth': [
                    'ApiKeyAuth',
                    'auth0Bearer'
                ],
                'endpoint_path': '/v1/datasets/{datasetId}/mappings',
                'operation_id': 'get_sample_mappings_by_dataset_id',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'dataset_id',
                    'field',
                ],
                'required': [
                    'dataset_id',
                    'field',
                ],
                'nullable': [
                ],
                'enum': [
                    'field',
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                    ('field',): {

                        "_ID": "_id",
                        "FILENAME": "fileName"
                    },
                },
                'openapi_types': {
                    'dataset_id':
                        (MongoObjectID,),
                    'field':
                        (str,),
                },
                'attribute_map': {
                    'dataset_id': 'datasetId',
                    'field': 'field',
                },
                'location_map': {
                    'dataset_id': 'path',
                    'field': 'query',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client,
            callable=__get_sample_mappings_by_dataset_id
        )
