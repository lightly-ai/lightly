import csv
from typing import List

from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData


class _UploadEmbeddingsMixin:

    def set_embedding_id_by_name(self, embedding_name: str = None):
        embeddings: List[DatasetEmbeddingData] = \
            self.embeddings_api.get_embeddings_by_dataset_id(dataset_id=self.dataset_id)

        if embedding_name is None:
            self.embedding_id = embeddings[-1].id
            return

        try:
            self.embedding_id = next(embedding.id for embedding in embeddings if embedding.name == embedding_name)
        except StopIteration:
            raise ValueError(f"No embedding with name {embedding_name} found on the server.")

    def upload_embeddings(self, path_to_embeddings_csv: str, name: str):
        """Uploads embeddings to the server.

        First checks that the specified embedding name is not on ther server. If it is, the upload is aborted.
        Then creates a new csv with the embeddings in the order specified on the server. Next it uploads it to the server.
        The received embedding_id is saved as a property of self.
        Args:
            path_to_embeddings_csv: the filepath to the .csv containing the embeddings, e.g. "path/to/embeddings.csv"
            name: The name of the embedding. If an embedding with such a name already exists on the server,
                the upload is aborted.

        Returns:
            None

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
            path_to_embeddings_csv=path_to_embeddings_csv)

        # get the URL to upload the csv to
        response: WriteCSVUrlData = \
            self.embeddings_api.get_embeddings_csv_write_url_by_id(self.dataset_id, name=name)
        self.embedding_id = response.embedding_id
        signed_write_url = response.signed_write_url

        # upload the csv to the URL
        with open(path_to_ordered_embeddings_csv, 'rb') as file_ordered_embeddings_csv:
            self.upload_file_with_signed_url(file=file_ordered_embeddings_csv, signed_write_url=signed_write_url)

    def _order_csv_by_filenames(self, path_to_embeddings_csv: str) -> str:
        """Orders the rows in a csv according to the order specified on the server and saves it as a new file.

        Args:
            path_to_embeddings_csv:
                the path to the csv to order

        Returns:
            the filepath to the new csv

        """
        with open(path_to_embeddings_csv, 'r') as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]
            index_filenames = header_row.index('filenames')
            filenames = [row[index_filenames] for row in rows_without_header]

            if len(filenames) != len(self.filenames_on_server):
                raise ValueError(f"There are {len(filenames)} rows in the embedding file, but "
                                 f"{len(self.filenames_on_server)} filenames/samples on the server.")
            if set(filenames) != set(self.filenames_on_server):
                raise ValueError(f"The filenames in the embedding file and the filenames on the server do not align")

            rows_without_header_ordered = self._order_list_by_filenames(filenames, rows_without_header)

            rows_to_write = [header_row]
            rows_to_write += rows_without_header_ordered

        path_to_ordered_embeddings_csv = path_to_embeddings_csv.replace('.csv', '_sorted.csv')
        with open(path_to_ordered_embeddings_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_write)

        return path_to_ordered_embeddings_csv
