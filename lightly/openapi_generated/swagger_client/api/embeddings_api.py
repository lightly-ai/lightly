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

from swagger_client.api_client import ApiClient


class EmbeddingsApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def get_embeddings_by_sample_id(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """get_embeddings_by_sample_id  # noqa: E501

        Get all embeddings of a datasets sample  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings_by_sample_id(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str mode: if we want everything (full) or just the summaries
        :return: list[EmbeddingData]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_embeddings_by_sample_id_with_http_info(dataset_id, sample_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_embeddings_by_sample_id_with_http_info(dataset_id, sample_id, **kwargs)  # noqa: E501
            return data

    def get_embeddings_by_sample_id_with_http_info(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """get_embeddings_by_sample_id  # noqa: E501

        Get all embeddings of a datasets sample  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings_by_sample_id_with_http_info(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str mode: if we want everything (full) or just the summaries
        :return: list[EmbeddingData]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'sample_id', 'mode']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_embeddings_by_sample_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_embeddings_by_sample_id`")  # noqa: E501
        # verify the required parameter 'sample_id' is set
        if ('sample_id' not in params or
                params['sample_id'] is None):
            raise ValueError("Missing the required parameter `sample_id` when calling `get_embeddings_by_sample_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'sample_id' in params:
            path_params['sampleId'] = params['sample_id']  # noqa: E501

        query_params = []
        if 'mode' in params:
            query_params.append(('mode', params['mode']))  # noqa: E501

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
            '/users/datasets/{datasetId}/samples/{sampleId}/embeddings', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='list[EmbeddingData]',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_embeddings_csv_write_url_by_id(self, dataset_id, **kwargs):  # noqa: E501
        """get_embeddings_csv_write_url_by_id  # noqa: E501

        Get the signed url to upload an CSVembedding to for a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings_csv_write_url_by_id(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param str name: the sampling requests name to create a signed url for
        :return: WriteCSVUrlData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_embeddings_csv_write_url_by_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_embeddings_csv_write_url_by_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
            return data

    def get_embeddings_csv_write_url_by_id_with_http_info(self, dataset_id, **kwargs):  # noqa: E501
        """get_embeddings_csv_write_url_by_id  # noqa: E501

        Get the signed url to upload an CSVembedding to for a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings_csv_write_url_by_id_with_http_info(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param str name: the sampling requests name to create a signed url for
        :return: WriteCSVUrlData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'name']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_embeddings_csv_write_url_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_embeddings_csv_write_url_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

        query_params = []
        if 'name' in params:
            query_params.append(('name', params['name']))  # noqa: E501

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
            '/v1/datasets/{datasetId}/embeddings/writeCSVUrl', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='WriteCSVUrlData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
