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


class TagsApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def create_initial_tag_by_dataset_id(self, body, dataset_id, **kwargs):  # noqa: E501
        """create_initial_tag_by_dataset_id  # noqa: E501

        create the intitial tag for a dataset which then locks the dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_initial_tag_by_dataset_id(body, dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param InitialTagCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.create_initial_tag_by_dataset_id_with_http_info(body, dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.create_initial_tag_by_dataset_id_with_http_info(body, dataset_id, **kwargs)  # noqa: E501
            return data

    def create_initial_tag_by_dataset_id_with_http_info(self, body, dataset_id, **kwargs):  # noqa: E501
        """create_initial_tag_by_dataset_id  # noqa: E501

        create the intitial tag for a dataset which then locks the dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_initial_tag_by_dataset_id_with_http_info(body, dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param InitialTagCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'dataset_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method create_initial_tag_by_dataset_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or
                params['body'] is None):
            raise ValueError("Missing the required parameter `body` when calling `create_initial_tag_by_dataset_id`")  # noqa: E501
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `create_initial_tag_by_dataset_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

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
            '/v1/datasets/{datasetId}/tags/initial', 'POST',
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

    def create_tag_by_dataset_id(self, body, dataset_id, **kwargs):  # noqa: E501
        """create_tag_by_dataset_id  # noqa: E501

        create new tag for dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_tag_by_dataset_id(body, dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param TagCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.create_tag_by_dataset_id_with_http_info(body, dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.create_tag_by_dataset_id_with_http_info(body, dataset_id, **kwargs)  # noqa: E501
            return data

    def create_tag_by_dataset_id_with_http_info(self, body, dataset_id, **kwargs):  # noqa: E501
        """create_tag_by_dataset_id  # noqa: E501

        create new tag for dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.create_tag_by_dataset_id_with_http_info(body, dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param TagCreateRequest body: (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: CreateEntityResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'dataset_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method create_tag_by_dataset_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if self.api_client.client_side_validation and ('body' not in params or
                                                       params['body'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `body` when calling `create_tag_by_dataset_id`")  # noqa: E501
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `create_tag_by_dataset_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

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
            '/v1/datasets/{datasetId}/tags', 'POST',
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

    def get_filenames_by_tag_id(self, dataset_id, tag_id, **kwargs):  # noqa: E501
        """get_filenames_by_tag_id  # noqa: E501

        Get list of filenames by tag  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_filenames_by_tag_id(dataset_id, tag_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID tag_id: ObjectId of the tag (required)
        :return: TagFilenamesData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_filenames_by_tag_id_with_http_info(dataset_id, tag_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_filenames_by_tag_id_with_http_info(dataset_id, tag_id, **kwargs)  # noqa: E501
            return data

    def get_filenames_by_tag_id_with_http_info(self, dataset_id, tag_id, **kwargs):  # noqa: E501
        """get_filenames_by_tag_id  # noqa: E501

        Get list of filenames by tag  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_filenames_by_tag_id_with_http_info(dataset_id, tag_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID tag_id: ObjectId of the tag (required)
        :return: TagFilenamesData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'tag_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_filenames_by_tag_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_filenames_by_tag_id`")  # noqa: E501
        # verify the required parameter 'tag_id' is set
        if ('tag_id' not in params or
                params['tag_id'] is None):
            raise ValueError("Missing the required parameter `tag_id` when calling `get_filenames_by_tag_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'tag_id' in params:
            path_params['tagId'] = params['tag_id']  # noqa: E501

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
            '/v1/datasets/{datasetId}/tags/{tagId}/filenames', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='TagFilenamesData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_tag_by_tag_id(self, dataset_id, tag_id, **kwargs):  # noqa: E501
        """get_tag_by_tag_id  # noqa: E501

        Get information about a specific tag  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_tag_by_tag_id(dataset_id, tag_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID tag_id: ObjectId of the tag (required)
        :return: TagData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_tag_by_tag_id_with_http_info(dataset_id, tag_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_tag_by_tag_id_with_http_info(dataset_id, tag_id, **kwargs)  # noqa: E501
            return data

    def get_tag_by_tag_id_with_http_info(self, dataset_id, tag_id, **kwargs):  # noqa: E501
        """get_tag_by_tag_id  # noqa: E501

        Get information about a specific tag  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_tag_by_tag_id_with_http_info(dataset_id, tag_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID tag_id: ObjectId of the tag (required)
        :return: TagData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'tag_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_tag_by_tag_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_tag_by_tag_id`")  # noqa: E501
        # verify the required parameter 'tag_id' is set
        if self.api_client.client_side_validation and ('tag_id' not in params or
                                                       params['tag_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `tag_id` when calling `get_tag_by_tag_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'tag_id' in params:
            path_params['tagId'] = params['tag_id']  # noqa: E501

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
            '/v1/datasets/{datasetId}/tags/{tagId}', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='TagData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_tags_by_dataset_id(self, dataset_id, **kwargs):  # noqa: E501
        """get_tags_by_dataset_id  # noqa: E501

        Get all tags of a dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_tags_by_dataset_id(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: list[TagData]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_tags_by_dataset_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_tags_by_dataset_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
            return data

    def get_tags_by_dataset_id_with_http_info(self, dataset_id, **kwargs):  # noqa: E501
        """get_tags_by_dataset_id  # noqa: E501

        Get all tags of a dataset  # noqa: E501
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_tags_by_dataset_id_with_http_info(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :return: list[TagData]
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
                    " to method get_tags_by_dataset_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if self.api_client.client_side_validation and ('dataset_id' not in params or
                                                       params['dataset_id'] is None):  # noqa: E501
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_tags_by_dataset_id`")  # noqa: E501

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
            '/v1/datasets/{datasetId}/tags', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='list[TagData]',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
