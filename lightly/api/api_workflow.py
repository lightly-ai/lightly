import time
from typing import *
import csv

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client import Configuration, ApiClient, SamplingsApi, JobsApi, JobState, \
    TagsApi, JobStatusData, EmbeddingsApi, MappingsApi, TagData
from lightly.api.upload import upload_file_with_signed_url
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData


class ApiWorkflow:
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

    def sampling(self, sampler_config: SamplerConfig, preselected_tag_id: str = None, query_tag_id: str = None,
                 al_scores: Dict[str, List[int]] = None) -> TagData:

        # upload the active learning scores to the api
        if al_scores is not None:
            raise NotImplementedError

        # trigger the sampling
        payload = sampler_config.get_as_api_sampling_create_request(
            preselected_tag_id=preselected_tag_id, query_tag_id=query_tag_id)
        payload.row_count = 15
        response = self.samplings_api.trigger_sampling_by_id(payload, self.dataset_id, self.embedding_id)
        job_id = response.job_id
        print(f"job_id: {job_id}")

        # poll the job status till the job is finished
        time.sleep(2)
        while True:
            job_status_data: JobStatusData = self.jobs_api.get_job_status_by_id(job_id=job_id)
            if job_status_data.status == JobState.FINISHED:
                new_tag = job_status_data.result.data
                break
            elif job_status_data.status == JobState.FAILED:
                raise ValueError(f"Sampling job failed with error {job_status_data.error}")

            time.sleep(job_status_data.wait_time_till_next_poll)

        # get the new tag from the job status
        new_tag_data = self.tags_api.get_tag_by_tag_id(self.dataset_id, tag_id=new_tag)

        return new_tag_data

    def upload_embeddings(self, path_to_embeddings_csv: str, name: str = None):

        # get the names of the current embeddings on the server:
        embeddings_on_server: List[DatasetEmbeddingData] = \
            self.embeddings_api.get_embeddings_by_dataset_id(dataset_id=self.dataset_id)
        names_embeddings_on_server = [embedding.name for embedding in embeddings_on_server]
        if name in names_embeddings_on_server:
            print(f"Aborting upload, embedding with name={name} already exists.")
            self.embedding_id = next(embedding for embedding in embeddings_on_server if embedding.name == name).id
            return

        # get the desired order of filenames
        filenames_on_server = self.mappings_api.get_sample_mappings_by_dataset_id(self.dataset_id, "fileName")

        # create a new csv with the filenames in the desired order
        path_to_ordered_embeddings_csv = self.__order_csv_by_filenames(path_to_embeddings_csv=path_to_embeddings_csv,
                                                                       filenames_in_desired_order=filenames_on_server,
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
                                 filenames_in_desired_order: List[str],
                                 pop_filename_column: bool = True
                                 ) -> str:

        with open(path_to_embeddings_csv, 'r') as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]
            index_filenames = header_row.index('filenames')
            row_dict = dict([(row[index_filenames], row) for row in rows_without_header])

            rows_to_write = [header_row]
            rows_to_write += [row_dict[filename] for filename in filenames_in_desired_order]

            if pop_filename_column:
                for row in rows_to_write:
                    row.pop(index_filenames)

        path_to_ordered_embeddings_csv = path_to_embeddings_csv.replace('.csv', '_sorted.csv')
        with open(path_to_ordered_embeddings_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

        return path_to_ordered_embeddings_csv
