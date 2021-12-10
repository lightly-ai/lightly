from typing import List
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import DatasourceRawSamplesData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import DatasourceRawSamplesDataRow
from lightly.openapi_generated.swagger_client.models.datasource_update_last_resource_request import DatasourceUpdateLastResourceRequest

import time

def _convert_raw_data_to_list(data: List[DatasourceRawSamplesDataRow]):
    return [(d.file_name, d.read_url) for d in data]


class _DatasourcesMixin:

    def download_raw_samples(self, dataset_id, _from: int = 0, to: int = None):
        if to is None:
            to = int(time.time())
        
        response: DatasourceRawSamplesData = self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id(
            dataset_id=dataset_id,
            _from=_from,
            to=to,
        )
        cursor = response.cursor
        data = _convert_raw_data_to_list(response.data)
        while response.has_more:
            response: DatasourceRawSamplesData = self._datasources_api.get_list_of_raw_samples_from_datasource_by_dataset_id(
                dataset_id=dataset_id,
                cursor=cursor
            )
            cursor = response.cursor
            new_data = _convert_raw_data_to_list(response.data)
            data.extend(new_data)
        return data

    def download_new_raw_samples(self, dataset_id):
        # get lastResourceAt
        response = self.last_resource_at(
            dataset_id=dataset_id
        )

        _from = int(response)
        to = int(time.time())
        data = self.download_raw_samples(dataset_id=dataset_id, _from=_from, to=to)

        # update lastResourceAt
        self.update_last_resource_at(dataset_id=dataset_id, timestamp=to)
        return data

    def last_resource_at(self, dataset_id: str):
        return self._datasources_api.get_last_resource_timestamp_of_datasource_by_dataset_id(
            dataset_id=dataset_id
        )

    def update_last_resource_at(self, dataset_id: str, timestamp: int):
        body = DatasourceUpdateLastResourceRequest(last_resource_at=timestamp)
        self._datasources_api.update_last_resource_timestamp_of_datasource_by_dataset_id(
            dataset_id=dataset_id,
            body=body
        )
