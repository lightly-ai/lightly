import time
from typing import *
import csv

from lightly.data.dataset import LightlyDataset

from lightly.api.upload import upload_file_with_signed_url, upload_images_from_folder, upload_dataset
from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import Configuration
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData


class ApiWorkflowClient:
    def __init__(self, host: str, token: str, dataset_id: str, embedding_id: str = None):

        configuration = Configuration()
        configuration.host = host
        configuration.api_key = {'token': token}
        api_client = ApiClient(configuration=configuration)
        self.api_client = api_client

        self.dataset_id = dataset_id
        if embedding_id is not None:
            self.embedding_id = embedding_id

        self.samplings_api = SamplingsApi(api_client=self.api_client)
        self.jobs_api = JobsApi(api_client=self.api_client)
        self.tags_api = TagsApi(api_client=self.api_client)
        self.embeddings_api = EmbeddingsApi(api_client=api_client)
        self.mappings_api = MappingsApi(api_client=api_client)

    def order_list_by_filenames(self, filenames_for_list: List[str], list_to_order: List[object]) -> List[object]:
        assert len(filenames_for_list) == len(list_to_order)
        dict_by_filenames = dict([(filename, element) for filename, element in zip(filenames_for_list, list_to_order)])
        list_ordered = [dict_by_filenames[filename] for filename in self.filenames_on_server
                        if filename in filenames_for_list]
        return list_ordered

    @property
    def filenames_on_server(self):
        if not hasattr(self, "_filenames"):
            self._filenames_on_server = self.mappings_api.\
                get_sample_mappings_by_dataset_id(dataset_id=self.dataset_id, field="fileName")
        return self._filenames_on_server

    def upload_dataset(self, input: Union[str, LightlyDataset], **kwargs):
        if isinstance(input, str):
            path_to_dataset = input
            upload_images_from_folder(path_to_dataset, self.dataset_id, self.token, **kwargs)
        elif isinstance(input, LightlyDataset):
            dataset = input
            upload_dataset(dataset, self.dataset_id, self.tags_api, **kwargs)
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")

    def upload_embeddings(self, path_to_embeddings_csv: str, name: str = None):

        # get the names of the current embeddings on the server:
        embeddings_on_server: List[DatasetEmbeddingData] = \
            self.embeddings_api.get_embeddings_by_dataset_id(dataset_id=self.dataset_id)
        names_embeddings_on_server = [embedding.name for embedding in embeddings_on_server]
        if name in names_embeddings_on_server:
            print(f"Aborting upload, embedding with name='{name}' already exists.")
            self.embedding_id = next(embedding for embedding in embeddings_on_server if embedding.name == name).id
            return

        # create a new csv with the filenames in the desired order
        path_to_ordered_embeddings_csv = self.__order_csv_by_filenames(path_to_embeddings_csv=path_to_embeddings_csv,
                                                                       pop_filename_column=True)

        # get the URL to upload the csv to
        response: WriteCSVUrlData = \
            self.embeddings_api.get_embeddings_csv_write_url_by_id(self.dataset_id, name=name)
        self.embedding_id = response.embedding_id
        signed_write_url = response.signed_write_url

        # upload the csv to the URL
        with open(path_to_ordered_embeddings_csv, 'rb') as file_ordered_embeddings_csv:
            upload_file_with_signed_url(file=file_ordered_embeddings_csv, url=signed_write_url)

    def __order_csv_by_filenames(self, path_to_embeddings_csv: str,
                                 pop_filename_column: bool = True
                                 ) -> str:

        with open(path_to_embeddings_csv, 'r') as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]
            index_filenames = header_row.index('filenames')
            filenames = [row[index_filenames] for row in rows_without_header]
            rows_without_header_ordered = self.order_list_by_filenames(filenames, rows_without_header)

            rows_to_write = [header_row]
            rows_to_write += rows_without_header_ordered

            if pop_filename_column:
                for row in rows_to_write:
                    row.pop(index_filenames)

        path_to_ordered_embeddings_csv = path_to_embeddings_csv.replace('.csv', '_sorted.csv')
        with open(path_to_ordered_embeddings_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

        return path_to_ordered_embeddings_csv
