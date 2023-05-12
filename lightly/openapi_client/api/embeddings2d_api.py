# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


import re  # noqa: F401
import io
import warnings

from pydantic import validate_arguments, ValidationError
from typing_extensions import Annotated

from pydantic import Field, constr, validator

from typing import List

from lightly.openapi_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_client.models.embedding2d_create_request import Embedding2dCreateRequest
from lightly.openapi_client.models.embedding2d_data import Embedding2dData

from lightly.openapi_client.api_client import ApiClient
from lightly.openapi_client.api_response import ApiResponse
from lightly.openapi_client.exceptions import (  # noqa: F401
    ApiTypeError,
    ApiValueError
)


class Embeddings2dApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient.get_default()
        self.api_client = api_client

    @validate_arguments
    def create_embeddings2d_by_embedding_id(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], embedding_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the embedding")], embedding2d_create_request : Embedding2dCreateRequest, **kwargs) -> CreateEntityResponse:  # noqa: E501
        """create_embeddings2d_by_embedding_id  # noqa: E501

        Create a new 2d embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.create_embeddings2d_by_embedding_id(dataset_id, embedding_id, embedding2d_create_request, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param embedding_id: ObjectId of the embedding (required)
        :type embedding_id: str
        :param embedding2d_create_request: (required)
        :type embedding2d_create_request: Embedding2dCreateRequest
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: CreateEntityResponse
        """
        kwargs['_return_http_data_only'] = True
        if '_preload_content' in kwargs:
            raise ValueError("Error! Please call the create_embeddings2d_by_embedding_id_with_http_info method with `_preload_content` instead and obtain raw data from ApiResponse.raw_data")
        return self.create_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, embedding2d_create_request, **kwargs)  # noqa: E501

    @validate_arguments
    def create_embeddings2d_by_embedding_id_with_http_info(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], embedding_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the embedding")], embedding2d_create_request : Embedding2dCreateRequest, **kwargs) -> ApiResponse:  # noqa: E501
        """create_embeddings2d_by_embedding_id  # noqa: E501

        Create a new 2d embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.create_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, embedding2d_create_request, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param embedding_id: ObjectId of the embedding (required)
        :type embedding_id: str
        :param embedding2d_create_request: (required)
        :type embedding2d_create_request: Embedding2dCreateRequest
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _preload_content: if False, the ApiResponse.data will
                                 be set to none and raw_data will store the 
                                 HTTP response body without reading/decoding.
                                 Default is True.
        :type _preload_content: bool, optional
        :param _return_http_data_only: response data instead of ApiResponse
                                       object with status code, headers, etc
        :type _return_http_data_only: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the authentication
                              in the spec for a single request.
        :type _request_auth: dict, optional
        :type _content_type: string, optional: force content-type for the request
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: tuple(CreateEntityResponse, status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'dataset_id',
            'embedding_id',
            'embedding2d_create_request'
        ]
        _all_params.extend(
            [
                'async_req',
                '_return_http_data_only',
                '_preload_content',
                '_request_timeout',
                '_request_auth',
                '_content_type',
                '_headers'
            ]
        )

        # validate the arguments
        for _key, _val in _params['kwargs'].items():
            if _key not in _all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method create_embeddings2d_by_embedding_id" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['dataset_id']:
            _path_params['datasetId'] = _params['dataset_id']

        if _params['embedding_id']:
            _path_params['embeddingId'] = _params['embedding_id']


        # process the query parameters
        _query_params = []
        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))
        # process the form parameters
        _form_params = []
        _files = {}
        # process the body parameter
        _body_params = None
        if _params['embedding2d_create_request'] is not None:
            _body_params = _params['embedding2d_create_request']

        # set the HTTP header `Accept`
        _header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # set the HTTP header `Content-Type`
        _content_types_list = _params.get('_content_type',
            self.api_client.select_header_content_type(
                ['application/json']))
        if _content_types_list:
                _header_params['Content-Type'] = _content_types_list

        # authentication setting
        _auth_settings = ['auth0Bearer', 'ApiKeyAuth']  # noqa: E501

        _response_types_map = {
            '201': "CreateEntityResponse",
            '400': "ApiErrorResponse",
            '401': "ApiErrorResponse",
            '403': "ApiErrorResponse",
            '404': "ApiErrorResponse",
        }

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d', 'POST',
            _path_params,
            _query_params,
            _header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            response_types_map=_response_types_map,
            auth_settings=_auth_settings,
            async_req=_params.get('async_req'),
            _return_http_data_only=_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=_params.get('_preload_content', True),
            _request_timeout=_params.get('_request_timeout'),
            collection_formats=_collection_formats,
            _request_auth=_params.get('_request_auth'))

    @validate_arguments
    def get_embedding2d_by_id(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], embedding_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the embedding")], embedding2d_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the 2d embedding")], **kwargs) -> Embedding2dData:  # noqa: E501
        """get_embedding2d_by_id  # noqa: E501

        Get the 2d embeddings by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_embedding2d_by_id(dataset_id, embedding_id, embedding2d_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param embedding_id: ObjectId of the embedding (required)
        :type embedding_id: str
        :param embedding2d_id: ObjectId of the 2d embedding (required)
        :type embedding2d_id: str
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: Embedding2dData
        """
        kwargs['_return_http_data_only'] = True
        if '_preload_content' in kwargs:
            raise ValueError("Error! Please call the get_embedding2d_by_id_with_http_info method with `_preload_content` instead and obtain raw data from ApiResponse.raw_data")
        return self.get_embedding2d_by_id_with_http_info(dataset_id, embedding_id, embedding2d_id, **kwargs)  # noqa: E501

    @validate_arguments
    def get_embedding2d_by_id_with_http_info(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], embedding_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the embedding")], embedding2d_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the 2d embedding")], **kwargs) -> ApiResponse:  # noqa: E501
        """get_embedding2d_by_id  # noqa: E501

        Get the 2d embeddings by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_embedding2d_by_id_with_http_info(dataset_id, embedding_id, embedding2d_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param embedding_id: ObjectId of the embedding (required)
        :type embedding_id: str
        :param embedding2d_id: ObjectId of the 2d embedding (required)
        :type embedding2d_id: str
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _preload_content: if False, the ApiResponse.data will
                                 be set to none and raw_data will store the 
                                 HTTP response body without reading/decoding.
                                 Default is True.
        :type _preload_content: bool, optional
        :param _return_http_data_only: response data instead of ApiResponse
                                       object with status code, headers, etc
        :type _return_http_data_only: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the authentication
                              in the spec for a single request.
        :type _request_auth: dict, optional
        :type _content_type: string, optional: force content-type for the request
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: tuple(Embedding2dData, status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'dataset_id',
            'embedding_id',
            'embedding2d_id'
        ]
        _all_params.extend(
            [
                'async_req',
                '_return_http_data_only',
                '_preload_content',
                '_request_timeout',
                '_request_auth',
                '_content_type',
                '_headers'
            ]
        )

        # validate the arguments
        for _key, _val in _params['kwargs'].items():
            if _key not in _all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_embedding2d_by_id" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['dataset_id']:
            _path_params['datasetId'] = _params['dataset_id']

        if _params['embedding_id']:
            _path_params['embeddingId'] = _params['embedding_id']

        if _params['embedding2d_id']:
            _path_params['embedding2dId'] = _params['embedding2d_id']


        # process the query parameters
        _query_params = []
        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))
        # process the form parameters
        _form_params = []
        _files = {}
        # process the body parameter
        _body_params = None
        # set the HTTP header `Accept`
        _header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # authentication setting
        _auth_settings = ['auth0Bearer', 'ApiKeyAuth']  # noqa: E501

        _response_types_map = {
            '200': "Embedding2dData",
            '400': "ApiErrorResponse",
            '401': "ApiErrorResponse",
            '403': "ApiErrorResponse",
            '404': "ApiErrorResponse",
        }

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d/{embedding2dId}', 'GET',
            _path_params,
            _query_params,
            _header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            response_types_map=_response_types_map,
            auth_settings=_auth_settings,
            async_req=_params.get('async_req'),
            _return_http_data_only=_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=_params.get('_preload_content', True),
            _request_timeout=_params.get('_request_timeout'),
            collection_formats=_collection_formats,
            _request_auth=_params.get('_request_auth'))

    @validate_arguments
    def get_embeddings2d_by_embedding_id(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], embedding_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the embedding")], **kwargs) -> List[Embedding2dData]:  # noqa: E501
        """get_embeddings2d_by_embedding_id  # noqa: E501

        Get all 2d embeddings of an embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_embeddings2d_by_embedding_id(dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param embedding_id: ObjectId of the embedding (required)
        :type embedding_id: str
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: List[Embedding2dData]
        """
        kwargs['_return_http_data_only'] = True
        if '_preload_content' in kwargs:
            raise ValueError("Error! Please call the get_embeddings2d_by_embedding_id_with_http_info method with `_preload_content` instead and obtain raw data from ApiResponse.raw_data")
        return self.get_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, **kwargs)  # noqa: E501

    @validate_arguments
    def get_embeddings2d_by_embedding_id_with_http_info(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], embedding_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the embedding")], **kwargs) -> ApiResponse:  # noqa: E501
        """get_embeddings2d_by_embedding_id  # noqa: E501

        Get all 2d embeddings of an embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param embedding_id: ObjectId of the embedding (required)
        :type embedding_id: str
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _preload_content: if False, the ApiResponse.data will
                                 be set to none and raw_data will store the 
                                 HTTP response body without reading/decoding.
                                 Default is True.
        :type _preload_content: bool, optional
        :param _return_http_data_only: response data instead of ApiResponse
                                       object with status code, headers, etc
        :type _return_http_data_only: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :param _request_auth: set to override the auth_settings for an a single
                              request; this effectively ignores the authentication
                              in the spec for a single request.
        :type _request_auth: dict, optional
        :type _content_type: string, optional: force content-type for the request
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: tuple(List[Embedding2dData], status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'dataset_id',
            'embedding_id'
        ]
        _all_params.extend(
            [
                'async_req',
                '_return_http_data_only',
                '_preload_content',
                '_request_timeout',
                '_request_auth',
                '_content_type',
                '_headers'
            ]
        )

        # validate the arguments
        for _key, _val in _params['kwargs'].items():
            if _key not in _all_params:
                raise ApiTypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_embeddings2d_by_embedding_id" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['dataset_id']:
            _path_params['datasetId'] = _params['dataset_id']

        if _params['embedding_id']:
            _path_params['embeddingId'] = _params['embedding_id']


        # process the query parameters
        _query_params = []
        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))
        # process the form parameters
        _form_params = []
        _files = {}
        # process the body parameter
        _body_params = None
        # set the HTTP header `Accept`
        _header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # authentication setting
        _auth_settings = ['auth0Bearer', 'ApiKeyAuth']  # noqa: E501

        _response_types_map = {
            '200': "List[Embedding2dData]",
            '400': "ApiErrorResponse",
            '401': "ApiErrorResponse",
            '403': "ApiErrorResponse",
            '404': "ApiErrorResponse",
        }

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d', 'GET',
            _path_params,
            _query_params,
            _header_params,
            body=_body_params,
            post_params=_form_params,
            files=_files,
            response_types_map=_response_types_map,
            auth_settings=_auth_settings,
            async_req=_params.get('async_req'),
            _return_http_data_only=_params.get('_return_http_data_only'),  # noqa: E501
            _preload_content=_params.get('_preload_content', True),
            _request_timeout=_params.get('_request_timeout'),
            collection_formats=_collection_formats,
            _request_auth=_params.get('_request_auth'))
