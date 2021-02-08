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


class DatasetsApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def get_dataset_by_id(self, dataset_id, **kwargs):  # noqa: E501
        """get_dataset_by_id  # noqa: E501

        Get a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_dataset_by_id(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: DatasetData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_dataset_by_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_dataset_by_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
            return data

    def get_dataset_by_id_with_http_info(self, dataset_id, **kwargs):  # noqa: E501
        """get_dataset_by_id  # noqa: E501

        Get a specific dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_dataset_by_id_with_http_info(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: DatasetData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_dataset_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_dataset_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

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
            '/users/datasets/{datasetId}', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='DatasetData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_datasets(self, **kwargs):  # noqa: E501
        """get_datasets  # noqa: E501

        Get all datasets for a user  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_datasets(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: list[DatasetData]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_datasets_with_http_info(**kwargs)  # noqa: E501
        else:
            (data) = self.get_datasets_with_http_info(**kwargs)  # noqa: E501
            return data

    def get_datasets_with_http_info(self, **kwargs):  # noqa: E501
        """get_datasets  # noqa: E501

        Get all datasets for a user  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_datasets_with_http_info(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :return: list[DatasetData]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = []  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_datasets" % key
                )
            params[key] = val
        del params['kwargs']

        collection_formats = {}

        path_params = {}

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
            '/users/datasets', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='list[DatasetData]',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
