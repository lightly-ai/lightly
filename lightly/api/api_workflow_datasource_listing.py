import time
import warnings
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import tqdm

from lightly.api.retry_utils import retry
from lightly.openapi_generated.swagger_client.models import (
    DatasourceConfig,
    DatasourceProcessedUntilTimestampRequest,
    DatasourceProcessedUntilTimestampResponse,
    DatasourceRawSamplesData,
)
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import (
    DatasourceRawSamplesDataRow,
)
from lightly.openapi_generated.swagger_client.models.divide_and_conquer_cursor_data import DivideAndConquerCursorData
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Union


DownloadFunction = Union[
    "DatasourcesApi.get_list_of_raw_samples_from_datasource_by_dataset_id",
    "DatasourcesApi.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id", 
    "DatasourcesApi.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
]

DivideAndConquerFunction = Union[
    "DatasourcesApi.get_divide_and_conquer_list_of_raw_samples_from_datasource_by_dataset_id",
    "DatasourcesApi.get_divide_and_conquer_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
    "DatasourcesApi.get_divide_and_conquer_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
]

class _DatasourceListingMixin:
    def download_raw_samples(
        self,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        divide_and_conquer_shards: int = 1,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads filenames and read urls from the datasource.

        Only samples with timestamp between `from_` (inclusive) and `to` (inclusive)
        will be downloaded.

        Args:
            from_:
                Unix timestamp from which on samples are downloaded. Defaults to the
                very beginning (timestamp 0).
            to:
                Unix timestamp up to and including which samples are downloaded.
                Defaults to the current timestamp.
            relevant_filenames_file_name:
                Path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root. Optional.
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.
            progress_bar:
                Tqdm progress bar to show how many samples have already been
                retrieved.

        Returns:
            A list of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.download_raw_samples()
            [('image-1.png', 'https://......'), ('image-2.png', 'https://......')]

        :meta private:  # Skip docstring generation
        """
        return self._download_raw_files(
            download_function=self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id,
            dnc_function=self._datasources_api.get_divide_and_conquer_list_of_raw_samples_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            divide_and_conquer_shards=divide_and_conquer_shards,
            progress_bar=progress_bar,
        )

    def download_raw_predictions(
        self,
        task_name: str,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        divide_and_conquer_shards: int = 1,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads all prediction filenames and read urls from the datasource.

        See `download_raw_predictions_iter` for details.

        :meta private:  # Skip docstring generation
        """
        return list(
            self.download_raw_predictions_iter(
                task_name=task_name,
                from_=from_,
                to=to,
                relevant_filenames_file_name=relevant_filenames_file_name,
                run_id=run_id,
                divide_and_conquer_shards=divide_and_conquer_shards,
                relevant_filenames_artifact_id=relevant_filenames_artifact_id,
                use_redirected_read_url=use_redirected_read_url,
                progress_bar=progress_bar,
            )
        )

    def download_raw_predictions_iter(
        self,
        task_name: str,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        divide_and_conquer_shards: int = 1,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> Iterator[Tuple[str, str]]:
        """Downloads prediction filenames and read urls from the datasource.

        Only samples with timestamp between `from_` (inclusive) and `to` (inclusive)
        will be downloaded.

        Args:
            task_name:
                Name of the prediction task.
            from_:
                Unix timestamp from which on samples are downloaded. Defaults to the
                very beginning (timestamp 0).
            to:
                Unix timestamp up to and including which samples are downloaded.
                Defaults to the current timestamp.
            relevant_filenames_file_name:
                Path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root. Optional.
            run_id:
                Run ID. Optional. Should be given along with
                `relevant_filenames_artifact_id` to download relevant files only.
            relevant_filenames_artifact_id:
                ID of the relevant filename artifact. Optional. Should be given along
                with `run_id` to download relevant files only. Note that this is
                different from `relevant_filenames_file_name`.
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.
            divide_and_conquer_shards:
                Number of shards to use for divide and conquer listing. Typically num_workers/cpu_count.
            progress_bar:
                Tqdm progress bar to show how many prediction files have already been
                retrieved.

        Returns:
            An iterator of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> task_name = "object-detection"
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> list(client.download_raw_predictions(task_name=task_name))
            [('.lightly/predictions/object-detection/image-1.json', 'https://......'),
             ('.lightly/predictions/object-detection/image-2.json', 'https://......')]

        :meta private:  # Skip docstring generation
        """
        if run_id is not None and relevant_filenames_artifact_id is None:
            raise ValueError(
                "'relevant_filenames_artifact_id' should not be `None` when 'run_id' "
                "is specified."
            )
        if run_id is None and relevant_filenames_artifact_id is not None:
            raise ValueError(
                "'run_id' should not be `None` when 'relevant_filenames_artifact_id' "
                "is specified."
            )
        relevant_filenames_kwargs = {}
        if run_id is not None and relevant_filenames_artifact_id is not None:
            relevant_filenames_kwargs["relevant_filenames_run_id"] = run_id
            relevant_filenames_kwargs[
                "relevant_filenames_artifact_id"
            ] = relevant_filenames_artifact_id

        cursors = self._get_divide_and_conquer_list_cursors(
            dnc_function=self._datasources_api.get_divide_and_conquer_list_of_raw_samples_predictions_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            use_redirected_read_url=use_redirected_read_url,
            task_name=task_name,
            divide_and_conquer_shards=divide_and_conquer_shards,
            **relevant_filenames_kwargs,
        )
        
        def download_with_cursor(cursor):
            return list(self._download_raw_files_cursor_iter(
                download_function=self._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id,
                cursor=cursor,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                task_name=task_name,
                progress_bar=progress_bar,
                **relevant_filenames_kwargs,
            ))

        # download in parallel using threads
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_with_cursor, cursor) for cursor in cursors]
            for future in as_completed(futures):
                yield from future.result()

    def download_raw_metadata(
        self,
        from_: int = 0,
        to: Optional[int] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        relevant_filenames_file_name: Optional[str] = None,
        divide_and_conquer_shards: int = 1,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Tuple[str, str]]:
        """Downloads all metadata filenames and read urls from the datasource.

        See `download_raw_metadata_iter` for details.

        :meta private:  # Skip docstring generation
        """
        return list(
            self.download_raw_metadata_iter(
                from_=from_,
                to=to,
                run_id=run_id,
                relevant_filenames_artifact_id=relevant_filenames_artifact_id,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                divide_and_conquer_shards=divide_and_conquer_shards,
                progress_bar=progress_bar,
            )
        )

    def download_raw_metadata_iter(
        self,
        from_: int = 0,
        to: Optional[int] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        divide_and_conquer_shards: int = 1,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> Iterator[Tuple[str, str]]:
        """Downloads all metadata filenames and read urls from the datasource.

        Only samples with timestamp between `from_` (inclusive) and `to` (inclusive)
        will be downloaded.

        Args:
            from_:
                Unix timestamp from which on samples are downloaded. Defaults to the
                very beginning (timestamp 0).
            to:
                Unix timestamp up to and including which samples are downloaded.
                Defaults to the current timestamp.
            relevant_filenames_file_name:
                Path to the relevant filenames text file in the cloud bucket.
                The path is relative to the datasource root. Optional.
            run_id:
                Run ID. Optional. Should be given along with
                `relevant_filenames_artifact_id` to download relevant files only.
            relevant_filenames_artifact_id:
                ID of the relevant filename artifact. Optional. Should be given along
                with `run_id` to download relevant files only. Note that this is
                different from `relevant_filenames_file_name`.
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.
            divide_and_conquer_shards:
                Number of shards to use for divide and conquer listing. Typically num_workers/cpu_count.
            progress_bar:
                Tqdm progress bar to show how many metadata files have already been
                retrieved.

        Returns:
            An iterator of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> list(client.download_raw_metadata_iter())
            [('.lightly/metadata/object-detection/image-1.json', 'https://......'),
             ('.lightly/metadata/object-detection/image-2.json', 'https://......')]

        :meta private:  # Skip docstring generation
        """
        if run_id is not None and relevant_filenames_artifact_id is None:
            raise ValueError(
                "'relevant_filenames_artifact_id' should not be `None` when 'run_id' "
                "is specified."
            )
        if run_id is None and relevant_filenames_artifact_id is not None:
            raise ValueError(
                "'run_id' should not be `None` when 'relevant_filenames_artifact_id' "
                "is specified."
            )
        relevant_filenames_kwargs = {}
        if run_id is not None and relevant_filenames_artifact_id is not None:
            relevant_filenames_kwargs["relevant_filenames_run_id"] = run_id
            relevant_filenames_kwargs[
                "relevant_filenames_artifact_id"
            ] = relevant_filenames_artifact_id

        cursors = self._get_divide_and_conquer_list_cursors(
            dnc_function=self._datasources_api.get_divide_and_conquer_list_of_raw_samples_metadata_from_datasource_by_dataset_id,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            divide_and_conquer_shards=divide_and_conquer_shards,
            **relevant_filenames_kwargs,
        )

        def download_with_cursor(cursor):
            return list(self._download_raw_files_cursor_iter(
                download_function=self._datasources_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id,
                cursor=cursor,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                progress_bar=progress_bar,
                **relevant_filenames_kwargs,
            ))

        # download in parallel using threads
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_with_cursor, cursor) for cursor in cursors]
            for future in as_completed(futures):
                yield from future.result()

    def download_new_raw_samples(
        self,
        use_redirected_read_url: bool = False,
    ) -> List[Tuple[str, str]]:
        """Downloads filenames and read urls of unprocessed samples from the datasource.

        All samples after the timestamp of `ApiWorkflowClient.get_processed_until_timestamp()` are
        fetched. After downloading the samples, the timestamp is updated to the current time.
        This function can be repeatedly called to retrieve new samples from the datasource.

        Args:
            use_redirected_read_url:
                Flag for redirected read urls. When this flag is true,
                RedirectedReadUrls are returned instead of ReadUrls, meaning that the
                returned URLs have unlimited access to the file.
                Defaults to False. When S3DelegatedAccess is configured, this flag has
                no effect because RedirectedReadUrls are always returned.

        Returns:
            A list of (filename, url) tuples where each tuple represents a sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.download_new_raw_samples()
            [('image-3.png', 'https://......'), ('image-4.png', 'https://......')]
        """
        from_ = self.get_processed_until_timestamp()

        if from_ != 0:
            # We already processed samples at some point.
            # Add 1 because the samples with timestamp == from_
            # have already been processed
            from_ += 1

        to = int(time.time())
        data = self.download_raw_samples(
            from_=from_,
            to=to,
            relevant_filenames_file_name=None,
            use_redirected_read_url=use_redirected_read_url,
        )
        self.update_processed_until_timestamp(timestamp=to)
        return data

    def get_processed_until_timestamp(self) -> int:
        """Returns the timestamp until which samples have been processed.

        Returns:
            Unix timestamp of last processed sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.get_processed_until_timestamp()
            1684750513

        :meta private:  # Skip docstring generation
        """
        response: DatasourceProcessedUntilTimestampResponse = retry(
            fn=self._datasources_api.get_datasource_processed_until_timestamp_by_dataset_id,
            dataset_id=self.dataset_id,
        )
        timestamp = int(response.processed_until_timestamp)
        return timestamp

    def update_processed_until_timestamp(self, timestamp: int) -> None:
        """Sets the timestamp until which samples have been processed.

        Args:
            timestamp:
                Unix timestamp of last processed sample.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset.
            >>> # All samples are processed at this moment.
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.download_new_raw_samples()
            []
            >>>
            >>> # Set timestamp to an earlier moment to reprocess samples
            >>> client.update_processed_until_timestamp(1684749813)
            >>> client.download_new_raw_samples()
            [('image-3.png', 'https://......'), ('image-4.png', 'https://......')]
        """
        body = DatasourceProcessedUntilTimestampRequest(
            processed_until_timestamp=timestamp
        )
        retry(
            fn=self._datasources_api.update_datasource_processed_until_timestamp_by_dataset_id,
            dataset_id=self.dataset_id,
            datasource_processed_until_timestamp_request=body,
        )

    def get_datasource(self) -> DatasourceConfig:
        """Returns the datasource of the current dataset.

        Returns:
            Datasource data of the datasource of the current dataset.

        Raises:
            ApiException if no datasource was configured.

        """
        return self._datasources_api.get_datasource_by_dataset_id(self.dataset_id)

    def get_prediction_read_url(
        self,
        filename: str,
    ) -> str:
        """Returns a read-url for .lightly/predictions/{filename}.

        Args:
            filename:
                Filename for which to get the read-url.

        Returns:
            A read-url to the file. Note that a URL will be returned even if the file does not
            exist.

        :meta private:  # Skip docstring generation
        """
        return self._datasources_api.get_prediction_file_read_url_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            file_name=filename,
        )

    def get_metadata_read_url(
        self,
        filename: str,
    ) -> str:
        """Returns a read-url for .lightly/metadata/{filename}.

        Args:
            filename:
                Filename for which to get the read-url.

        Returns:
            A read-url to the file. Note that a URL will be returned even if the file does not
            exist.

        :meta private:  # Skip docstring generation
        """
        return self._datasources_api.get_metadata_file_read_url_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            file_name=filename,
        )

    def get_custom_embedding_read_url(
        self,
        filename: str,
    ) -> str:
        """Returns a read-url for .lightly/embeddings/{filename}.

        Args:
            filename:
                Filename for which to get the read-url.

        Returns:
            A read-url to the file. Note that a URL will be returned even if the file does not
            exist.

        :meta private:  # Skip docstring generation
        """
        return self._datasources_api.get_custom_embedding_file_read_url_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            file_name=filename,
        )

    
    def _get_divide_and_conquer_list_cursors(
        self,
        dnc_function: DivideAndConquerFunction,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        divide_and_conquer_shards: int = 1,
        **kwargs,
    ) -> List[str]:
        
        if to is None:
            to = int(time.time())

        divide_and_conquer_shards = max(1, divide_and_conquer_shards)

        relevant_filenames_kwargs = (
            {"relevant_filenames_file_name": relevant_filenames_file_name}
            if relevant_filenames_file_name
            else dict()
        )
        response: DivideAndConquerCursorData = retry(
            fn=dnc_function,
            dataset_id=self.dataset_id,
            var_from=from_,
            to=to,
            dnc_shards=divide_and_conquer_shards,
            **relevant_filenames_kwargs,
            **kwargs,
        )

        return response.cursors
    
    def _download_raw_files(
        self,
        download_function: DownloadFunction,
        dnc_function: DivideAndConquerFunction,
        from_: int = 0,
        to: Optional[int] = None,
        relevant_filenames_file_name: Optional[str] = None,
        divide_and_conquer_shards: int = 1,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ) -> List[Tuple[str, str]]:
    
        return list(
            self._download_raw_files_divide_and_conquer_iter(
                download_function=download_function,
                dnc_function=dnc_function,
                from_=from_,
                to=to,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                divide_and_conquer_shards=divide_and_conquer_shards,
                progress_bar=progress_bar,
                **kwargs,
            )
        )

    def _download_raw_files_divide_and_conquer_iter(
        self,
        download_function: DownloadFunction,
        dnc_function: DivideAndConquerFunction,
        from_: int = 0,
        to: Optional[int] = None,
        run_id: Optional[str] = None,
        relevant_filenames_artifact_id: Optional[str] = None,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        divide_and_conquer_shards: int = 1,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ) -> Iterator[Tuple[str, str]]:
        if run_id is not None and relevant_filenames_artifact_id is None:
            raise ValueError(
                "'relevant_filenames_artifact_id' should not be `None` when 'run_id' "
                "is specified."
            )
        if run_id is None and relevant_filenames_artifact_id is not None:
            raise ValueError(
                "'run_id' should not be `None` when 'relevant_filenames_artifact_id' "
                "is specified."
            )
        relevant_filenames_kwargs = {}
        if run_id is not None and relevant_filenames_artifact_id is not None:
            relevant_filenames_kwargs["relevant_filenames_run_id"] = run_id
            relevant_filenames_kwargs[
                "relevant_filenames_artifact_id"
            ] = relevant_filenames_artifact_id

        cursors = self._get_divide_and_conquer_list_cursors(
            dnc_function=dnc_function,
            from_=from_,
            to=to,
            relevant_filenames_file_name=relevant_filenames_file_name,
            divide_and_conquer_shards=divide_and_conquer_shards,
            **relevant_filenames_kwargs,
        )

        def download_with_cursor(cursor):
            return list(self._download_raw_files_cursor_iter(
                download_function=download_function,
                cursor=cursor,
                relevant_filenames_file_name=relevant_filenames_file_name,
                use_redirected_read_url=use_redirected_read_url,
                progress_bar=progress_bar,
                **relevant_filenames_kwargs,
            ))

        # download in parallel using threads
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_with_cursor, cursor) for cursor in cursors]
            for future in as_completed(futures):
                yield from future.result()


    def _download_raw_files_cursor_iter(
        self,
        download_function: DownloadFunction,
        cursor: str,
        relevant_filenames_file_name: Optional[str] = None,
        use_redirected_read_url: bool = False,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ) -> Iterator[Tuple[str, str]]:
        relevant_filenames_kwargs = (
            {"relevant_filenames_file_name": relevant_filenames_file_name}
            if relevant_filenames_file_name
            else dict()
        )

        listed_filenames = set()

        def get_entries(
            response: DatasourceRawSamplesData,
        ) -> Iterator[Tuple[str, str]]:
            for entry in response.data:
                if _sample_unseen_and_valid(
                    sample=entry,
                    relevant_filenames_file_name=relevant_filenames_file_name,
                    listed_filenames=listed_filenames,
                ):
                    listed_filenames.add(entry.file_name)
                    yield entry.file_name, entry.read_url
                if progress_bar is not None:
                    progress_bar.update(1)

        active_cursor = cursor 
        while active_cursor:
            print('calling..')
            response: DatasourceRawSamplesData = retry(
                fn=download_function,
                dataset_id=self.dataset_id,
                cursor=active_cursor,
                use_redirected_read_url=use_redirected_read_url,
                **relevant_filenames_kwargs,
                **kwargs,
            )
            print(f"Downloading samples with cursor: {active_cursor} {len(response.data)} samples found")
            yield from get_entries(response=response)

            active_cursor = response.cursor
            if not response.has_more:
                active_cursor = None


def _sample_unseen_and_valid(
    sample: DatasourceRawSamplesDataRow,
    relevant_filenames_file_name: Optional[str],
    listed_filenames: Set[str],
) -> bool:
    # Note: We want to remove these checks eventually. Absolute paths and relative paths
    # with dot notation should be handled either in the API or the Worker. Duplicate
    # filenames should be handled in the Worker as handling it in the API would require
    # too much memory.
    if sample.file_name.startswith("/"):
        warnings.warn(
            UserWarning(
                f"Absolute file paths like {sample.file_name} are not supported"
                f" in relevant filenames file {relevant_filenames_file_name} due to blob storage"
            )
        )
        return False
    elif sample.file_name.startswith(("./", "../")):
        warnings.warn(
            UserWarning(
                f"Using dot notation ('./', '../') like in {sample.file_name} is not supported"
                f" in relevant filenames file {relevant_filenames_file_name} due to blob storage"
            )
        )
        return False
    elif sample.file_name in listed_filenames:
        warnings.warn(
            UserWarning(
                f"Duplicate filename {sample.file_name} in relevant"
                f" filenames file {relevant_filenames_file_name}"
            )
        )
        return False
    else:
        return True
