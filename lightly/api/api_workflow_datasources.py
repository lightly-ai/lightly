from typing import List
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import (
    DatasourceRawSamplesData,
)
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import (
    DatasourceRawSamplesDataRow,
)
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_request import (
    DatasourceProcessedUntilTimestampRequest,
)

import time

def _convert_raw_data_to_list(data: List[DatasourceRawSamplesDataRow]):
    return [(d.file_name, d.read_url) for d in data]


class _DatasourcesMixin:

    def download_raw_samples(self, _from: int = 0, to: int = None):
        if to is None:
            to = int(time.time())
        response: DatasourceRawSamplesData = self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id(
            dataset_id=self.dataset_id,
            _from=_from,
            to=to,
        )
        cursor = response.cursor
        data = _convert_raw_data_to_list(response.data)
        while response.has_more:
            response: DatasourceRawSamplesData = self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id(
                dataset_id=self.dataset_id, cursor=cursor
            )
            cursor = response.cursor
            new_data = _convert_raw_data_to_list(response.data)
            data.extend(new_data)
        return data

    def download_new_raw_samples(self):
        response = self.get_processed_until_timestamp()
        _from = int(response)
        
        if _from != 0:
            # We already processed samples at some point.
            # Add 1 because the samples with timestamp == _from
            # have already been processed
            _from += 1

        to = int(time.time())
        data = self.download_raw_samples(_from=_from, to=to)
        self.update_processed_until_timestamp(timestamp=to)
        return data

    def get_processed_until_timestamp(self):
        return self._datasources_api.get_datasource_processed_until_timestamp_by_dataset_id(
            dataset_id=self.dataset_id
        )

    def update_processed_until_timestamp(self, timestamp: int):
        body = DatasourceProcessedUntilTimestampRequest(
            processed_until_timestamp=timestamp
        )
        self._datasources_api.update_datasource_processed_until_timestamp_by_dataset_id(
            dataset_id=self.dataset_id, body=body
        )
