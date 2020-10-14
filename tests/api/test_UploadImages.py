import unittest
import json
import os

import random
import responses

import torchvision
import tempfile
import lightly.api as api


class TestUploadImages(unittest.TestCase):

    def setup(self, n_data=1000):

        # set up url
        self.dataset_id = 'XYZ'
        self.token = 'secret'
        self.dst_url = os.getenvb(b'LIGHTLY_SERVER_LOCATION',
                                  b'https://api.lightly.ai').decode()

        self.gettag_url = f'{self.dst_url}/users/datasets/{self.dataset_id}/tags/?token={self.token}'
        self.sample_url = f'{self.dst_url}/users/datasets/{self.dataset_id}/samples/'
        self.signed_url = 'https://www.this-is-a-signed-url.com'
        self.dataset_url = f'{self.dst_url}/users/datasets/{self.dataset_id}?token={self.token}'
        self.maketag_url = f'{self.dst_url}/users/datasets/{self.dataset_id}/tags'
        self.getquota_url = f'{self.dst_url}/users/quota'

        # create a dataset
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)


    @responses.activate
    def test_upload_images_dataset_too_large(self):
        self.setup(n_data=25001)

        def get_quota_callback(request):
            return (
                200,
                [],
                json.dumps({'maxDatasetSize': 25000})
            )
        responses.add_callback(
            responses.GET, self.getquota_url,
            callback=get_quota_callback,
            content_type='application/json'
        )

        with self.assertRaises(ValueError):
            api.upload_images_from_folder(
                self.folder_path,
                dataset_id=self.dataset_id,
                token=self.token
            )

    @responses.activate
    def test_upload_images_tag_exists(self):
        self.setup(n_data=10)

        def get_tags_callback(request):
            return (
                200,
                [],
                json.dumps([{'name': 'initial-tag'}, {'name': 'other-tag'}])
            )
        
        def get_quota_callback(request):
            return (
                200,
                [],
                json.dumps({'maxDatasetSize': 25000})
            )

        responses.add_callback(
            responses.GET, self.gettag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.getquota_url,
            callback=get_quota_callback,
            content_type='application/json'
        )

        with self.assertRaises(RuntimeError):
            api.upload_images_from_folder(
                self.folder_path,
                dataset_id=self.dataset_id,
                token=self.token
            )

    @responses.activate
    def test_upload_images_metadata(self):
        self.setup(n_data=10)
        
        def get_tags_callback(request):
            return (200,[], json.dumps([]))
        
        def upload_sample_callback(request):
            return (200, [], json.dumps({'sampleId': 'x1y2'}))
        
        def put_dataset_callback(request):
            return (200,[], json.dumps([]))
        
        def post_tag_callback(request):
            return (200, [], json.dumps([]))

        def get_quota_callback(request):
            return (
                200,
                [],
                json.dumps({'maxDatasetSize': 25000})
            )

        responses.add_callback(
            responses.GET, self.gettag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.POST, self.sample_url,
            callback=upload_sample_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.PUT, self.dataset_url,
            callback=put_dataset_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.POST, self.maketag_url,
            callback=post_tag_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.getquota_url,
            callback=get_quota_callback,
            content_type='application/json'
        )

        api.upload_images_from_folder(
            self.folder_path,
            dataset_id=self.dataset_id,
            token=self.token,
            mode='metadata'
        )

    @responses.activate
    def test_upload_images_thumbnails(self):
        self.setup(n_data=10)
        
        def get_tags_callback(request):
            return (200,[], json.dumps([]))
        
        def upload_sample_callback(request):
            return (200, [], json.dumps({'sampleId': 'x1y2'}))
        
        def get_thumbnail_write_url_callback(request):
            return (200, [], json.dumps({'signedWriteUrl': self.signed_url}))
        
        def put_thumbnail_callback(request):
            return (200, [], json.dumps({}))
        
        def put_dataset_callback(request):
            return (200,[], json.dumps([]))
        
        def post_tag_callback(request):
            return (200, [], json.dumps([]))

        def get_quota_callback(request):
            return (
                200,
                [],
                json.dumps({'maxDatasetSize': 25000})
            )

        responses.add_callback(
            responses.GET, self.gettag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.POST, self.sample_url,
            callback=upload_sample_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET,
            f'{self.dst_url}/users/datasets/{self.dataset_id}/samples/x1y2/writeurl',
            callback=get_thumbnail_write_url_callback,
            content_type='application/json'
        )
        
        responses.add_callback(
            responses.PUT, self.signed_url,
            callback=put_thumbnail_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.PUT, self.dataset_url,
            callback=put_dataset_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.POST, self.maketag_url,
            callback=post_tag_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.getquota_url,
            callback=get_quota_callback,
            content_type='application/json'
        )

        api.upload_images_from_folder(
            self.folder_path,
            dataset_id=self.dataset_id,
            token=self.token,
            mode='thumbnails'
        )

    @responses.activate
    def test_upload_images_full(self):
        self.setup(n_data=10)
        
        def get_tags_callback(request):
            return (200,[], json.dumps([]))
        
        def upload_sample_callback(request):
            return (200, [], json.dumps({'sampleId': 'x1y2'}))
        
        def get_thumbnail_write_url_callback(request):
            return (200, [], json.dumps({'signedWriteUrl': self.signed_url}))
        
        def put_thumbnail_callback(request):
            return (200, [], json.dumps({}))
        
        def put_dataset_callback(request):
            return (200,[], json.dumps([]))
        
        def post_tag_callback(request):
            return (200, [], json.dumps([]))

        def get_quota_callback(request):
            return (
                200,
                [],
                json.dumps({'maxDatasetSize': 25000})
            )

        responses.add_callback(
            responses.GET, self.gettag_url,
            callback=get_tags_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.POST, self.sample_url,
            callback=upload_sample_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET,
            f'{self.dst_url}/users/datasets/{self.dataset_id}/samples/x1y2/writeurl',
            callback=get_thumbnail_write_url_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.PUT, self.signed_url,
            callback=put_thumbnail_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.PUT, self.dataset_url,
            callback=put_dataset_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.POST, self.maketag_url,
            callback=post_tag_callback,
            content_type='application/json'
        )

        responses.add_callback(
            responses.GET, self.getquota_url,
            callback=get_quota_callback,
            content_type='application/json'
        )

        api.upload_images_from_folder(
            self.folder_path,
            dataset_id=self.dataset_id,
            token=self.token,
            mode='full'
        )
