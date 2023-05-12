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

from lightly.openapi_client.models.active_learning_score_create_request import ActiveLearningScoreCreateRequest
from lightly.openapi_client.models.active_learning_score_data import ActiveLearningScoreData
from lightly.openapi_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_client.models.tag_active_learning_scores_data import TagActiveLearningScoresData

from lightly.openapi_client.api_client import ApiClient
from lightly.openapi_client.api_response import ApiResponse
from lightly.openapi_client.exceptions import (  # noqa: F401
    ApiTypeError,
    ApiValueError
)


class ScoresApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient.get_default()
        self.api_client = api_client

    @validate_arguments
    def create_or_update_active_learning_score_by_tag_id(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], tag_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the tag")], active_learning_score_create_request : ActiveLearningScoreCreateRequest, **kwargs) -> CreateEntityResponse:  # noqa: E501
        """create_or_update_active_learning_score_by_tag_id  # noqa: E501

        Create or update active learning score object by tag id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.create_or_update_active_learning_score_by_tag_id(dataset_id, tag_id, active_learning_score_create_request, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param tag_id: ObjectId of the tag (required)
        :type tag_id: str
        :param active_learning_score_create_request: (required)
        :type active_learning_score_create_request: ActiveLearningScoreCreateRequest
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
            raise ValueError("Error! Please call the create_or_update_active_learning_score_by_tag_id_with_http_info method with `_preload_content` instead and obtain raw data from ApiResponse.raw_data")
        return self.create_or_update_active_learning_score_by_tag_id_with_http_info(dataset_id, tag_id, active_learning_score_create_request, **kwargs)  # noqa: E501

    @validate_arguments
    def create_or_update_active_learning_score_by_tag_id_with_http_info(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], tag_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the tag")], active_learning_score_create_request : ActiveLearningScoreCreateRequest, **kwargs) -> ApiResponse:  # noqa: E501
        """create_or_update_active_learning_score_by_tag_id  # noqa: E501

        Create or update active learning score object by tag id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.create_or_update_active_learning_score_by_tag_id_with_http_info(dataset_id, tag_id, active_learning_score_create_request, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param tag_id: ObjectId of the tag (required)
        :type tag_id: str
        :param active_learning_score_create_request: (required)
        :type active_learning_score_create_request: ActiveLearningScoreCreateRequest
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
            'tag_id',
            'active_learning_score_create_request'
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
                    " to method create_or_update_active_learning_score_by_tag_id" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['dataset_id']:
            _path_params['datasetId'] = _params['dataset_id']

        if _params['tag_id']:
            _path_params['tagId'] = _params['tag_id']


        # process the query parameters
        _query_params = []
        # process the header parameters
        _header_params = dict(_params.get('_headers', {}))
        # process the form parameters
        _form_params = []
        _files = {}
        # process the body parameter
        _body_params = None
        if _params['active_learning_score_create_request'] is not None:
            _body_params = _params['active_learning_score_create_request']

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
            '/v1/datasets/{datasetId}/tags/{tagId}/scores', 'POST',
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
    def get_active_learning_score_by_score_id(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], tag_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the tag")], score_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the scores")], **kwargs) -> ActiveLearningScoreData:  # noqa: E501
        """get_active_learning_score_by_score_id  # noqa: E501

        Get active learning score object by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_active_learning_score_by_score_id(dataset_id, tag_id, score_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param tag_id: ObjectId of the tag (required)
        :type tag_id: str
        :param score_id: ObjectId of the scores (required)
        :type score_id: str
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: ActiveLearningScoreData
        """
        kwargs['_return_http_data_only'] = True
        if '_preload_content' in kwargs:
            raise ValueError("Error! Please call the get_active_learning_score_by_score_id_with_http_info method with `_preload_content` instead and obtain raw data from ApiResponse.raw_data")
        return self.get_active_learning_score_by_score_id_with_http_info(dataset_id, tag_id, score_id, **kwargs)  # noqa: E501

    @validate_arguments
    def get_active_learning_score_by_score_id_with_http_info(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], tag_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the tag")], score_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the scores")], **kwargs) -> ApiResponse:  # noqa: E501
        """get_active_learning_score_by_score_id  # noqa: E501

        Get active learning score object by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_active_learning_score_by_score_id_with_http_info(dataset_id, tag_id, score_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param tag_id: ObjectId of the tag (required)
        :type tag_id: str
        :param score_id: ObjectId of the scores (required)
        :type score_id: str
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
        :rtype: tuple(ActiveLearningScoreData, status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'dataset_id',
            'tag_id',
            'score_id'
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
                    " to method get_active_learning_score_by_score_id" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['dataset_id']:
            _path_params['datasetId'] = _params['dataset_id']

        if _params['tag_id']:
            _path_params['tagId'] = _params['tag_id']

        if _params['score_id']:
            _path_params['scoreId'] = _params['score_id']


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
            '200': "ActiveLearningScoreData",
            '400': "ApiErrorResponse",
            '401': "ApiErrorResponse",
            '403': "ApiErrorResponse",
            '404': "ApiErrorResponse",
        }

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/tags/{tagId}/scores/{scoreId}', 'GET',
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
    def get_active_learning_scores_by_tag_id(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], tag_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the tag")], **kwargs) -> List[TagActiveLearningScoresData]:  # noqa: E501
        """get_active_learning_scores_by_tag_id  # noqa: E501

        Get all scoreIds for the given tag  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_active_learning_scores_by_tag_id(dataset_id, tag_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param tag_id: ObjectId of the tag (required)
        :type tag_id: str
        :param async_req: Whether to execute the request asynchronously.
        :type async_req: bool, optional
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        :return: Returns the result object.
                 If the method is called asynchronously,
                 returns the request thread.
        :rtype: List[TagActiveLearningScoresData]
        """
        kwargs['_return_http_data_only'] = True
        if '_preload_content' in kwargs:
            raise ValueError("Error! Please call the get_active_learning_scores_by_tag_id_with_http_info method with `_preload_content` instead and obtain raw data from ApiResponse.raw_data")
        return self.get_active_learning_scores_by_tag_id_with_http_info(dataset_id, tag_id, **kwargs)  # noqa: E501

    @validate_arguments
    def get_active_learning_scores_by_tag_id_with_http_info(self, dataset_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the dataset")], tag_id : Annotated[constr(strict=True), Field(..., description="ObjectId of the tag")], **kwargs) -> ApiResponse:  # noqa: E501
        """get_active_learning_scores_by_tag_id  # noqa: E501

        Get all scoreIds for the given tag  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.get_active_learning_scores_by_tag_id_with_http_info(dataset_id, tag_id, async_req=True)
        >>> result = thread.get()

        :param dataset_id: ObjectId of the dataset (required)
        :type dataset_id: str
        :param tag_id: ObjectId of the tag (required)
        :type tag_id: str
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
        :rtype: tuple(List[TagActiveLearningScoresData], status_code(int), headers(HTTPHeaderDict))
        """

        _params = locals()

        _all_params = [
            'dataset_id',
            'tag_id'
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
                    " to method get_active_learning_scores_by_tag_id" % _key
                )
            _params[_key] = _val
        del _params['kwargs']

        _collection_formats = {}

        # process the path parameters
        _path_params = {}
        if _params['dataset_id']:
            _path_params['datasetId'] = _params['dataset_id']

        if _params['tag_id']:
            _path_params['tagId'] = _params['tag_id']


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
            '200': "List[TagActiveLearningScoresData]",
            '400': "ApiErrorResponse",
            '401': "ApiErrorResponse",
            '403': "ApiErrorResponse",
            '404': "ApiErrorResponse",
        }

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/tags/{tagId}/scores', 'GET',
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
