import csv
import io
import tempfile
import urllib.request
from datetime import datetime
from typing import List
from urllib.request import Request

from lightly.api.utils import retry
from lightly.openapi_generated.swagger_client.models import (
    DatasetEmbeddingData,
    DimensionalityReductionMethod,
    Trigger2dEmbeddingJobRequest,
    WriteCSVUrlData,
)
from lightly.utils import io as io_utils


class EmbeddingDoesNotExistError(ValueError):
    pass


class _UploadEmbeddingsMixin:
    def _get_csv_reader_from_read_url(self, read_url: str) -> None:
        """Makes a get request to the signed read url and returns the .csv file."""
        request = Request(read_url, method="GET")
        with urllib.request.urlopen(request) as response:
            buffer = io.StringIO(response.read().decode("utf-8"))
            reader = csv.reader(buffer)

        return reader

    def set_embedding_id_to_latest(self) -> None:
        """Sets the embedding ID in the API client to the latest embedding ID in the current dataset.

        :meta private:  # Skip docstring generation
        """
        embeddings_on_server: List[
            DatasetEmbeddingData
        ] = self._embeddings_api.get_embeddings_by_dataset_id(
            dataset_id=self.dataset_id
        )
        if len(embeddings_on_server) == 0:
            raise RuntimeError(
                f"There are no known embeddings for dataset_id {self.dataset_id}."
            )
        # return first entry as the API returns newest first
        self.embedding_id = embeddings_on_server[0].id

    def get_embedding_by_name(
        self, name: str, ignore_suffix: bool = True
    ) -> DatasetEmbeddingData:
        """Fetches an embedding in the current dataset by name.

        Args:
            name:
                The name of the desired embedding.
            ignore_suffix:
                If true, a suffix of the embedding name in the current dataset
                is ignored.

        Returns:
            The embedding data.

        Raises:
            EmbeddingDoesNotExistError:
                If the name does not match the name of an embedding
                on the server.

        """
        embeddings_on_server: List[
            DatasetEmbeddingData
        ] = self._embeddings_api.get_embeddings_by_dataset_id(
            dataset_id=self.dataset_id
        )
        try:
            if ignore_suffix:
                embedding = next(
                    embedding
                    for embedding in embeddings_on_server
                    if embedding.name.startswith(name)
                )
            else:
                embedding = next(
                    embedding
                    for embedding in embeddings_on_server
                    if embedding.name == name
                )
        except StopIteration:
            raise EmbeddingDoesNotExistError(
                f"Embedding with the specified name "
                f"does not exist on the server: {name}"
            )
        return embedding

    def upload_embeddings(self, path_to_embeddings_csv: str, name: str) -> None:
        """Uploads embeddings to the Lightly Platform.

        First checks that the specified embedding name is not on the server. If it is, the upload is aborted.
        Then creates a new csv file with the embeddings in the order specified on the server. Next uploads it
        to the Lightly Platform. The received embedding ID is stored in the API client.

        Args:
            path_to_embeddings_csv:
                The path to the .csv containing the embeddings, e.g. "path/to/embeddings.csv"
            name:
                The name of the embedding. If an embedding with such a name already exists on the server,
                the upload is aborted.

        :meta private:  # Skip docstring generation
        """
        io_utils.check_embeddings(
            path_to_embeddings_csv, remove_additional_columns=True
        )

        # Try to append the embeddings on the server, if they exist
        try:
            embedding = self.get_embedding_by_name(name, ignore_suffix=True)
            # -> append rows from server
            print("Appending embeddings from server.")
            self.append_embeddings(path_to_embeddings_csv, embedding.id)
            now = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
            name = f"{name}_{now}"
        except EmbeddingDoesNotExistError:
            pass

        # create a new csv with the filenames in the desired order
        rows_csv = self._order_csv_by_filenames(
            path_to_embeddings_csv=path_to_embeddings_csv
        )

        # get the URL to upload the csv to
        response: WriteCSVUrlData = (
            self._embeddings_api.get_embeddings_csv_write_url_by_id(
                self.dataset_id, name=name
            )
        )
        self.embedding_id = response.embedding_id
        signed_write_url = response.signed_write_url

        # save the csv rows in a temporary in-memory string file
        # using a csv writer and then read them as bytes
        with tempfile.SpooledTemporaryFile(mode="rw") as f:
            writer = csv.writer(f)
            writer.writerows(rows_csv)
            f.seek(0)
            embeddings_csv_as_bytes = f.read().encode("utf-8")

        # write the bytes to a temporary in-memory byte file
        with tempfile.SpooledTemporaryFile(mode="r+b") as f_bytes:
            f_bytes.write(embeddings_csv_as_bytes)
            f_bytes.seek(0)
            retry(
                self.upload_file_with_signed_url,
                file=f_bytes,
                signed_write_url=signed_write_url,
            )

        # trigger the 2d embeddings job
        for dimensionality_reduction_method in [
            DimensionalityReductionMethod.PCA,
            DimensionalityReductionMethod.TSNE,
            DimensionalityReductionMethod.UMAP,
        ]:
            body = Trigger2dEmbeddingJobRequest(
                dimensionality_reduction_method=dimensionality_reduction_method
            )
            self._embeddings_api.trigger2d_embeddings_job(
                trigger2d_embedding_job_request=body,
                dataset_id=self.dataset_id,
                embedding_id=self.embedding_id,
            )

    def append_embeddings(self, path_to_embeddings_csv: str, embedding_id: str) -> None:
        """Concatenates embeddings from the Lightly Platform to the local ones.

        Loads the embedding csv file with the corresponding embedding ID in the current dataset
        and appends all of its rows to the local embeddings file located at
        'path_to_embeddings_csv'.

        Args:
            path_to_embeddings_csv:
                The path to the csv containing the local embeddings.
            embedding_id:
                ID of the embedding summary of the embeddings on the Lightly Platform.

        Raises:
            RuntimeError:
                If the number of columns in the local embeddings file and that of the remote
                embeddings file mismatch.

        :meta private:  # Skip docstring generation
        """

        # read embedding from API
        embedding_read_url = self._embeddings_api.get_embeddings_csv_read_url_by_id(
            self.dataset_id, embedding_id
        )
        embedding_reader = self._get_csv_reader_from_read_url(embedding_read_url)
        rows = list(embedding_reader)
        header, online_rows = rows[0], rows[1:]

        # read local embedding
        with open(path_to_embeddings_csv, "r") as f:
            local_rows = list(csv.reader(f))

            if len(local_rows[0]) != len(header):
                raise RuntimeError(
                    "Column mismatch! Number of columns in local and remote"
                    f" embeddings files must match but are {len(local_rows[0])}"
                    f" and {len(header)} respectively."
                )

            local_rows = local_rows[1:]

        # combine online and local embeddings
        total_rows = [header]
        filename_to_local_row = {row[0]: row for row in local_rows}
        for row in online_rows:
            # pick local over online filename if it exists
            total_rows.append(filename_to_local_row.pop(row[0], row))
        # add all local rows which were not added yet
        total_rows.extend(list(filename_to_local_row.values()))

        # save embeddings again
        with open(path_to_embeddings_csv, "w") as f:
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
        with open(path_to_embeddings_csv, "r") as f:
            data = csv.reader(f)

            rows = list(data)
            header_row = rows[0]
            rows_without_header = rows[1:]
            index_filenames = header_row.index("filenames")
            filenames = [row[index_filenames] for row in rows_without_header]

            filenames_on_server = self.get_filenames()

            if len(filenames) != len(filenames_on_server):
                raise ValueError(
                    f"There are {len(filenames)} rows in the embedding file, but "
                    f"{len(filenames_on_server)} filenames/samples on the server."
                )
            if set(filenames) != set(filenames_on_server):
                raise ValueError(
                    f"The filenames in the embedding file and "
                    f"the filenames on the server do not align"
                )

            rows_without_header_ordered = self._order_list_by_filenames(
                filenames, rows_without_header
            )

            rows_csv = [header_row]
            rows_csv += rows_without_header_ordered

        return rows_csv
