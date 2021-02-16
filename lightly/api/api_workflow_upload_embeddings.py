from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

import csv
from typing import List

from lightly.api.upload import upload_file_with_signed_url
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData


class _UploadEmbeddingsMixin:
    def upload_embeddings(self: ApiWorkflowClient, path_to_embeddings_csv: str, name: str):
        """
        Uploads embedding to the server. The received embedding_id is saved as a property of self.
        Args:
            path_to_embeddings_csv: the filepath to the .csv containing the embeddings, e.g. "path/to/embeddings.csv"
            name: The name of the embedding. If an embedding with such a name already exists on the server,
                the upload is aborted.

        Returns: None

        """
        # get the names of the current embeddings on the server:
        embeddings_on_server: List[DatasetEmbeddingData] = \
            self.embeddings_api.get_embeddings_by_dataset_id(dataset_id=self.dataset_id)
        names_embeddings_on_server = [embedding.name for embedding in embeddings_on_server]
        if name in names_embeddings_on_server:
            print(f"Aborting upload, embedding with name='{name}' already exists.")
            self.embedding_id = next(embedding for embedding in embeddings_on_server if embedding.name == name).id
            return

        # create a new csv with the filenames in the desired order
        path_to_ordered_embeddings_csv = self._order_csv_by_filenames(
            path_to_embeddings_csv=path_to_embeddings_csv, pop_filename_column=True)

        # get the URL to upload the csv to
        response: WriteCSVUrlData = \
            self.embeddings_api.get_embeddings_csv_write_url_by_id(self.dataset_id, name=name)
        self.embedding_id = response.embedding_id
        signed_write_url = response.signed_write_url

        # upload the csv to the URL
        with open(path_to_ordered_embeddings_csv, 'rb') as file_ordered_embeddings_csv:
            upload_file_with_signed_url(file=file_ordered_embeddings_csv, url=signed_write_url)

    def _order_csv_by_filenames(self: ApiWorkflowClient, path_to_embeddings_csv: str,
                                pop_filename_column: bool = True
                                ) -> str:
        """
        Orders the rows in a csv according to the order specified on the server and saves it as a new file
        Args:
            path_to_embeddings_csv: the path to the csv to order
            pop_filename_column: if set to True, the filename column is removed,
            as the order of filenames is already specified on the server.

        Returns: the filepath to the new csv

        """
        with open(path_to_embeddings_csv, 'r') as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]
            index_filenames = header_row.index('filenames')
            filenames = [row[index_filenames] for row in rows_without_header]
            rows_without_header_ordered = self._order_list_by_filenames(filenames, rows_without_header)

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
