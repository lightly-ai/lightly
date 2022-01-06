import io
import csv
import tempfile
import hashlib
from datetime import datetime
from typing import List
from urllib.request import Request, urlopen

from lightly.openapi_generated.swagger_client import \
    DimensionalityReductionMethod, Trigger2dEmbeddingJobRequest
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data \
    import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data \
    import WriteCSVUrlData
from lightly.utils.io import check_filenames


class EmbeddingDoesNotExistError(ValueError):
    pass


class _UploadEmbeddingsMixin:

    def _get_csv_reader_from_read_url(self, read_url: str):
        """Makes a get request to the signed read url and returns the .csv file.

        """
        request = Request(read_url, method='GET')
        with urlopen(request) as response:
            buffer = io.StringIO(response.read().decode('utf-8'))
            reader = csv.reader(buffer)

        return reader

    def set_embedding_id_to_latest(self):
        """Sets the self.embedding_id to the one of the latest on the server.

        """
        embeddings_on_server: List[DatasetEmbeddingData] = \
            self._embeddings_api.get_embeddings_by_dataset_id(
                dataset_id=self.dataset_id
            )
        self.embedding_id = embeddings_on_server[-1].id

    def get_embedding_by_name(
            self, name: str, ignore_suffix: bool = True
    ) -> DatasetEmbeddingData:
        """Gets an embedding form the server by name.

        Args:
            name:
                The name of the embedding to get.
            ignore_suffix:
                If true, a suffix of the embedding name on the server
                is ignored.

        Returns:
            The embedding data.

        Raises:
            EmbeddingDoesNotExistError:
                If the name does not match the name of an embedding
                on the server.

        """
        embeddings_on_server: List[DatasetEmbeddingData] = \
            self._embeddings_api.get_embeddings_by_dataset_id(
                dataset_id=self.dataset_id
            )
        try:
            if ignore_suffix:
                embedding = next(
                    embedding for embedding in embeddings_on_server if
                    embedding.name.startswith(name)
                )
            else:
                embedding = next(
                    embedding for embedding in embeddings_on_server if
                    embedding.name == name
                )
        except StopIteration:
            raise EmbeddingDoesNotExistError(
                f"Embedding with the specified name "
                f"does not exist on the server: {name}"
            )
        return embedding

    def upload_embeddings(self, path_to_embeddings_csv: str, name: str):
        """Uploads embeddings to the server.

        First checks that the specified embedding name is not on ther server. If it is, the upload is aborted.
        Then creates a new csv with the embeddings in the order specified on the server. Next it uploads it to the server.
        The received embedding_id is saved as a property of self.

        Args:
            path_to_embeddings_csv:
                The path to the .csv containing the embeddings, e.g. "path/to/embeddings.csv"
            name:
                The name of the embedding. If an embedding with such a name already exists on the server,
                the upload is aborted.

        """

        # Try to append the embeddings on the server, if they exist
        try:
            embedding = self.get_embedding_by_name(name, ignore_suffix=True)
            # -> append rows from server
            print('Appending embeddings from server.')
            self.append_embeddings(path_to_embeddings_csv, embedding.id)
            now = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')
            name = f'{name}_{now}'
        except EmbeddingDoesNotExistError:
            pass

        # create a new csv with the filenames in the desired order
        rows_csv = self._order_csv_by_filenames(
            path_to_embeddings_csv=path_to_embeddings_csv)

        # get the URL to upload the csv to
        response: WriteCSVUrlData = \
            self._embeddings_api.get_embeddings_csv_write_url_by_id(self.dataset_id, name=name)
        self.embedding_id = response.embedding_id
        signed_write_url = response.signed_write_url

        # save the csv rows in a temporary in-memory string file
        # using a csv writer and then read them as bytes
        with tempfile.SpooledTemporaryFile(mode="rw") as f:
            writer = csv.writer(f)
            writer.writerows(rows_csv)
            f.seek(0)
            embeddings_csv_as_bytes = f.read().encode('utf-8')

        # write the bytes to a temporary in-memory byte file
        with tempfile.SpooledTemporaryFile(mode='r+b') as f_bytes:
            f_bytes.write(embeddings_csv_as_bytes)
            f_bytes.seek(0)
            self.upload_file_with_signed_url(file=f_bytes, signed_write_url=signed_write_url)

        # trigger the 2d embeddings job
        for dimensionality_reduction_method in [
            DimensionalityReductionMethod.PCA,
            DimensionalityReductionMethod.TSNE,
            DimensionalityReductionMethod.UMAP
        ]:

            body = Trigger2dEmbeddingJobRequest(
                dimensionality_reduction_method=dimensionality_reduction_method)
            self._embeddings_api.trigger2d_embeddings_job(
                body=body,
                dataset_id=self.dataset_id,
                embedding_id=self.embedding_id
            )


    def append_embeddings(self,
                          path_to_embeddings_csv: str,
                          embedding_id: str):
        """Concatenates the embeddings from the server to the local ones.

        Loads the embedding csv file belonging to the embedding_id, and
        appends all of its rows to the local embeddings file located at
        'path_to_embeddings_csv'.

        Args:
            path_to_embeddings_csv:
                The path to the csv containing the local embeddings.
            embedding_id:
                Id of the embedding summary of the embeddings on the server.

        Raises:
            RuntimeError:
                If the number of columns in the local and the remote
                embeddings file mismatch.
        
        """

        # read embedding from API
        embedding_read_url = self._embeddings_api \
            .get_embeddings_csv_read_url_by_id(self.dataset_id, embedding_id)
        embedding_reader = self._get_csv_reader_from_read_url(embedding_read_url)
        rows = list(embedding_reader)
        header, online_rows = rows[0], rows[1:]

        # read local embedding
        with open(path_to_embeddings_csv, 'r') as f:
            local_rows = list(csv.reader(f))

            if len(local_rows[0]) != len(header):
                raise RuntimeError(
                    'Column mismatch! Number of columns in local and remote'
                    f' embeddings files must match but are {len(local_rows[0])}'
                    f' and {len(header)} respectively.'
                )

            local_rows = local_rows[1:]

        # combine online and local embeddings
        total_rows = [header]
        filename_to_local_row = { row[0]: row for row in local_rows }
        for row in online_rows:
            # pick local over online filename if it exists
            total_rows.append(filename_to_local_row.pop(row[0], row))
        # add all local rows which were not added yet
        total_rows.extend(list(filename_to_local_row.values()))
        
        # save embeddings again
        with open(path_to_embeddings_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(total_rows)
        

    def _order_csv_by_filenames(self, path_to_embeddings_csv: str) -> List[str]:
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

            filenames_on_server = self.get_filenames()

            if len(filenames) != len(filenames_on_server):
                raise ValueError(f'There are {len(filenames)} rows in the embedding file, but '
                                 f'{len(filenames_on_server)} filenames/samples on the server.')
            if set(filenames) != set(filenames_on_server):
                raise ValueError(f'The filenames in the embedding file and '
                                 f'the filenames on the server do not align')
            check_filenames(filenames)

            rows_without_header_ordered = \
                self._order_list_by_filenames(filenames, rows_without_header)

            rows_csv = [header_row]
            rows_csv += rows_without_header_ordered

        return rows_csv
