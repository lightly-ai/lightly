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


class SamplesApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def get_embeddings_by_sample_id(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """Get all embeddings of a datasets sample  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings_by_sample_id(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str mode: if we want everything (full) or just the summaries
        :return: InlineResponse2001
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
        """Get all embeddings of a datasets sample  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_embeddings_by_sample_id_with_http_info(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str mode: if we want everything (full) or just the summaries
        :return: InlineResponse2001
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
        auth_settings = ['auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/users/datasets/{datasetId}/samples/{sampleId}/embeddings', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='InlineResponse2001',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_sample_by_id(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """Get a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_sample_by_id(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :return: SampleData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_sample_by_id_with_http_info(dataset_id, sample_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_sample_by_id_with_http_info(dataset_id, sample_id, **kwargs)  # noqa: E501
            return data

    def get_sample_by_id_with_http_info(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """Get a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_sample_by_id_with_http_info(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :return: SampleData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'sample_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_sample_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_sample_by_id`")  # noqa: E501
        # verify the required parameter 'sample_id' is set
        if ('sample_id' not in params or
                params['sample_id'] is None):
            raise ValueError("Missing the required parameter `sample_id` when calling `get_sample_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'sample_id' in params:
            path_params['sampleId'] = params['sample_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/users/datasets/{datasetId}/samples/{sampleId}', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='SampleData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_sample_image_read_url_by_id(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """Get the image path of a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_sample_image_read_url_by_id(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str type: if we want to get the full image or just the thumbnail
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_sample_image_read_url_by_id_with_http_info(dataset_id, sample_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_sample_image_read_url_by_id_with_http_info(dataset_id, sample_id, **kwargs)  # noqa: E501
            return data

    def get_sample_image_read_url_by_id_with_http_info(self, dataset_id, sample_id, **kwargs):  # noqa: E501
        """Get the image path of a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_sample_image_read_url_by_id_with_http_info(dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str type: if we want to get the full image or just the thumbnail
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'sample_id', 'type']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_sample_image_read_url_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_sample_image_read_url_by_id`")  # noqa: E501
        # verify the required parameter 'sample_id' is set
        if ('sample_id' not in params or
                params['sample_id'] is None):
            raise ValueError("Missing the required parameter `sample_id` when calling `get_sample_image_read_url_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'sample_id' in params:
            path_params['sampleId'] = params['sample_id']  # noqa: E501

        query_params = []
        if 'type' in params:
            query_params.append(('type', params['type']))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/users/datasets/{datasetId}/samples/{sampleId}/readurl', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='str',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_sample_image_write_url_by_id(self, dataset_id, sample_id, file_name, **kwargs):  # noqa: E501
        """Get the signed url to upload an image to for a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_sample_image_write_url_by_id(dataset_id, sample_id, file_name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str file_name: the filename to create a signed url for (required)
        :return: InlineResponse200
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_sample_image_write_url_by_id_with_http_info(dataset_id, sample_id, file_name, **kwargs)  # noqa: E501
        else:
            (data) = self.get_sample_image_write_url_by_id_with_http_info(dataset_id, sample_id, file_name, **kwargs)  # noqa: E501
            return data

    def get_sample_image_write_url_by_id_with_http_info(self, dataset_id, sample_id, file_name, **kwargs):  # noqa: E501
        """Get the signed url to upload an image to for a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_sample_image_write_url_by_id_with_http_info(dataset_id, sample_id, file_name, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :param str file_name: the filename to create a signed url for (required)
        :return: InlineResponse200
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'sample_id', 'file_name']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_sample_image_write_url_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_sample_image_write_url_by_id`")  # noqa: E501
        # verify the required parameter 'sample_id' is set
        if ('sample_id' not in params or
                params['sample_id'] is None):
            raise ValueError("Missing the required parameter `sample_id` when calling `get_sample_image_write_url_by_id`")  # noqa: E501
        # verify the required parameter 'file_name' is set
        if ('file_name' not in params or
                params['file_name'] is None):
            raise ValueError("Missing the required parameter `file_name` when calling `get_sample_image_write_url_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'sample_id' in params:
            path_params['sampleId'] = params['sample_id']  # noqa: E501

        query_params = []
        if 'file_name' in params:
            query_params.append(('fileName', params['file_name']))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/users/datasets/{datasetId}/samples/{sampleId}/writeurl', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='InlineResponse200',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def get_samples_by_dataset_id(self, dataset_id, **kwargs):  # noqa: E501
        """Get all samples of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_samples_by_dataset_id(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param str mode: if we want everything (full) or just the ObjectIds
        :param str filename: filter the samples by filename
        :return: list[SampleData]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.get_samples_by_dataset_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
        else:
            (data) = self.get_samples_by_dataset_id_with_http_info(dataset_id, **kwargs)  # noqa: E501
            return data

    def get_samples_by_dataset_id_with_http_info(self, dataset_id, **kwargs):  # noqa: E501
        """Get all samples of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.get_samples_by_dataset_id_with_http_info(dataset_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param str mode: if we want everything (full) or just the ObjectIds
        :param str filename: filter the samples by filename
        :return: list[SampleData]
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['dataset_id', 'mode', 'filename']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method get_samples_by_dataset_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `get_samples_by_dataset_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501

        query_params = []
        if 'mode' in params:
            query_params.append(('mode', params['mode']))  # noqa: E501
        if 'filename' in params:
            query_params.append(('filename', params['filename']))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/users/datasets/{datasetId}/samples', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='list[SampleData]',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def update_sample_by_id(self, body, dataset_id, sample_id, **kwargs):  # noqa: E501
        """update a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.update_sample_by_id(body, dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Body body: the updated sample to set (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :return: SampleData
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.update_sample_by_id_with_http_info(body, dataset_id, sample_id, **kwargs)  # noqa: E501
        else:
            (data) = self.update_sample_by_id_with_http_info(body, dataset_id, sample_id, **kwargs)  # noqa: E501
            return data

    def update_sample_by_id_with_http_info(self, body, dataset_id, sample_id, **kwargs):  # noqa: E501
        """update a specific sample of a dataset  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.update_sample_by_id_with_http_info(body, dataset_id, sample_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param Body body: the updated sample to set (required)
        :param MongoObjectID dataset_id: ObjectId of the dataset (required)
        :param MongoObjectID sample_id: ObjectId of the sample (required)
        :return: SampleData
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['body', 'dataset_id', 'sample_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method update_sample_by_id" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'body' is set
        if ('body' not in params or
                params['body'] is None):
            raise ValueError("Missing the required parameter `body` when calling `update_sample_by_id`")  # noqa: E501
        # verify the required parameter 'dataset_id' is set
        if ('dataset_id' not in params or
                params['dataset_id'] is None):
            raise ValueError("Missing the required parameter `dataset_id` when calling `update_sample_by_id`")  # noqa: E501
        # verify the required parameter 'sample_id' is set
        if ('sample_id' not in params or
                params['sample_id'] is None):
            raise ValueError("Missing the required parameter `sample_id` when calling `update_sample_by_id`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'dataset_id' in params:
            path_params['datasetId'] = params['dataset_id']  # noqa: E501
        if 'sample_id' in params:
            path_params['sampleId'] = params['sample_id']  # noqa: E501

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
        auth_settings = ['auth0Bearer']  # noqa: E501

        return self.api_client.call_api(
            '/users/datasets/{datasetId}/samples/{sampleId}', 'PUT',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='SampleData',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)
