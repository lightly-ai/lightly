# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six

from lightly.openapi_generated.swagger_client.api_client import ApiClient


class Embeddings2dApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def create_embeddings2d_by_embedding_id(self, body, dataset_id, embedding_id, **kwargs):  # noqa: E501
        """create_embeddings2d_by_embedding_id  # noqa: E501

        Create a new 2d embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_embeddings2d_by_embedding_id(body, dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Embedding2dCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID embedding_id: ObjectId of the embedding (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.create_embeddings2d_by_embedding_id_with_http_info(body, dataset_id, embedding_id, **kwargs)  # noqa: E501
        else:
            (data) = self.create_embeddings2d_by_embedding_id_with_http_info(body, dataset_id, embedding_id, **kwargs)  # noqa: E501
            return data

    def create_embeddings2d_by_embedding_id_with_http_info(self, body, dataset_id, embedding_id, **kwargs):  # noqa: E501
        """create_embeddings2d_by_embedding_id  # noqa: E501

        Create a new 2d embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_embeddings2d_by_embedding_id_with_http_info(body, dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Embedding2dCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID embedding_id: ObjectId of the embedding (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'dataset_id', 'embedding_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method create_embeddings2d_by_embedding_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if self.api_client.client_side_validation and ('body' not in params or
                                                       params['body'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `body` when calling `create_embeddings2d_by_embedding_id`")  # noqa: E501
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `create_embeddings2d_by_embedding_id`")  # noqa: E501
        # verify the required parameter 'embedding_id' is set
        if self.api_client.client_side_validation and ('embedding_id' not in params or
                                                       params['embedding_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `embedding_id` when calling `create_embeddings2d_by_embedding_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'embedding_id' in params:
            path_params['embeddingId'] = params['embedding_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        if 'body' in params:
            body_params = params['body']
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth', 'auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d', 'POST',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='CreateEntityResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_embedding2d_by_id(self, dataset_id, embedding_id, embedding2d_id, **kwargs):  # noqa: E501
        """get_embedding2d_by_id  # noqa: E501

        Get the 2d embeddings by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embedding2d_by_id(dataset_id, embedding_id, embedding2d_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID embedding_id: ObjectId of the embedding (required)
        :param MongoObjectID embedding2d_id: ObjectId of the 2d embedding (required)
        :return: Embedding2dData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_embedding2d_by_id_with_http_info(dataset_id, embedding_id, embedding2d_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_embedding2d_by_id_with_http_info(dataset_id, embedding_id, embedding2d_id, **kwargs)  # noqa: E501
            return data

    def get_embedding2d_by_id_with_http_info(self, dataset_id, embedding_id, embedding2d_id, **kwargs):  # noqa: E501
        """get_embedding2d_by_id  # noqa: E501

        Get the 2d embeddings by id  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embedding2d_by_id_with_http_info(dataset_id, embedding_id, embedding2d_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID embedding_id: ObjectId of the embedding (required)
        :param MongoObjectID embedding2d_id: ObjectId of the 2d embedding (required)
        :return: Embedding2dData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'embedding_id', 'embedding2d_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_embedding2d_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_embedding2d_by_id`")  # noqa: E501
        # verify the required parameter 'embedding_id' is set
        if self.api_client.client_side_validation and ('embedding_id' not in params or
                                                       params['embedding_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `embedding_id` when calling `get_embedding2d_by_id`")  # noqa: E501
        # verify the required parameter 'embedding2d_id' is set
        if self.api_client.client_side_validation and ('embedding2d_id' not in params or
                                                       params['embedding2d_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `embedding2d_id` when calling `get_embedding2d_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'embedding_id' in params:
            path_params['embeddingId'] = params['embedding_id']  # noqa: E501
        if 'embedding2d_id' in params:
            path_params['embedding2dId'] = params['embedding2d_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth', 'auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d/{embedding2dId}', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='Embedding2dData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_embeddings2d_by_embedding_id(self, dataset_id, embedding_id, **kwargs):  # noqa: E501
        """get_embeddings2d_by_embedding_id  # noqa: E501

        Get all 2d embeddings of an embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings2d_by_embedding_id(dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID embedding_id: ObjectId of the embedding (required)
        :return: list[Embedding2dData]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, **kwargs)  # noqa: E501
            return data

    def get_embeddings2d_by_embedding_id_with_http_info(self, dataset_id, embedding_id, **kwargs):  # noqa: E501
        """get_embeddings2d_by_embedding_id  # noqa: E501

        Get all 2d embeddings of an embedding  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings2d_by_embedding_id_with_http_info(dataset_id, embedding_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID embedding_id: ObjectId of the embedding (required)
        :return: list[Embedding2dData]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'embedding_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_embeddings2d_by_embedding_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_embeddings2d_by_embedding_id`")  # noqa: E501
        # verify the required parameter 'embedding_id' is set
        if self.api_client.client_side_validation and ('embedding_id' not in params or
                                                       params['embedding_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `embedding_id` when calling `get_embeddings2d_by_embedding_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'embedding_id' in params:
            path_params['embeddingId'] = params['embedding_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['ApiKeyAuth', 'auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/v1/datasets/{datasetId}/embeddings/{embeddingId}/2d', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='list[Embedding2dData]',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
