import warnings
from typing import Dict, List

from lightly.api import utils
from lightly.openapi_generated.swagger_client.models import (
    FileNameFormat,
    LabelBoxDataRow,
    LabelBoxV4DataRow,
    LabelStudioTask,
)


class _ExportDatasetMixin:
    def export_label_studio_tasks_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict]:
        """Fetches samples in a format compatible with Label Studio.

        The format is documented here:
        https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format

        More information:
        https://docs.lightly.ai/docs/labelstudio-integration

        Args:
            tag_id:
                ID of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Label Studio.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.export_label_studio_tasks_by_tag_id(tag_id="646f34608a5613b57d8b73cc")
            [{'id': 0, 'data': {'image': '...', ...}}]
        """
        label_studio_tasks: List[LabelStudioTask] = list(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_label_studio_tasks,
                page_size=20000,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
            )
        )
        return [task.to_dict(by_alias=True) for task in label_studio_tasks]

    def export_label_studio_tasks_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict]:
        """Fetches samples in a format compatible with Label Studio.

        The format is documented here:
        https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format

        More information:
        https://docs.lightly.ai/docs/labelstudio-integration

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Label Studio.

        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> tasks = client.export_label_studio_tasks_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>>
            >>> with open('my-label-studio-tasks.json', 'w') as f:
            >>>     json.dump(tasks, f)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_label_studio_tasks_by_tag_id(tag.id)

    def export_label_box_data_rows_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict]:
        """Fetches samples in a format compatible with Labelbox v3.

        The format is documented here: https://docs.labelbox.com/docs/images-json

        More information:
        https://docs.lightly.ai/docs/labelbox

        Args:
            tag_id:
                ID of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Labelbox v3.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.export_label_box_data_rows_by_tag_id(tag_id="646f34608a5613b57d8b73cc")
            [{'externalId': '2218961434_7916358f53_z.jpg', 'imageUrl': ...}]
        """
        warnings.warn(
            DeprecationWarning(
                "This method exports data in the deprecated Labelbox v3 format and "
                "will be removed in the future. Use export_label_box_v4_data_rows_by_tag_id "
                "to export data in the Labelbox v4 format instead."
            )
        )
        label_box_data_rows: List[LabelBoxDataRow] = list(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_label_box_data_rows,
                page_size=20000,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
            )
        )
        return [row.to_dict(by_alias=True) for row in label_box_data_rows]

    def export_label_box_data_rows_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict]:
        """Fetches samples in a format compatible with Labelbox v3.

        The format is documented here: https://docs.labelbox.com/docs/images-json

        More information:
        https://docs.lightly.ai/docs/labelbox

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Labelbox v3.

        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> tasks = client.export_label_box_data_rows_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>>
            >>> with open('my-labelbox-rows.json', 'w') as f:
            >>>     json.dump(tasks, f)

        """
        warnings.warn(
            DeprecationWarning(
                "This method exports data in the deprecated Labelbox v3 format and "
                "will be removed in the future. Use export_label_box_v4_data_rows_by_tag_name "
                "to export data in the Labelbox v4 format instead."
            )
        )
        tag = self.get_tag_by_name(tag_name)
        return self.export_label_box_data_rows_by_tag_id(tag.id)

    def export_label_box_v4_data_rows_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict]:
        """Fetches samples in a format compatible with Labelbox v4.

        The format is documented here: https://docs.labelbox.com/docs/images-json

        More information:
        https://docs.lightly.ai/docs/labelbox

        Args:
            tag_id:
                ID of the tag which should exported.
        Returns:
            A list of dictionaries in a format compatible with Labelbox v4.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.export_label_box_v4_data_rows_by_tag_id(tag_id="646f34608a5613b57d8b73cc")
            [{'row_data': '...', 'global_key': 'image-1.jpg', 'media_type': 'IMAGE'}
        """
        label_box_data_rows: List[LabelBoxV4DataRow] = list(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_label_box_v4_data_rows,
                page_size=20000,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
            )
        )
        return [row.to_dict() for row in label_box_data_rows]

    def export_label_box_v4_data_rows_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict]:
        """Fetches samples in a format compatible with Labelbox.

        The format is documented here: https://docs.labelbox.com/docs/images-json

        More information:
        https://docs.lightly.ai/docs/labelbox

        Args:
            tag_name:
                Name of the tag which should exported.
        Returns:
            A list of dictionaries in a format compatible with Labelbox.
        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> tasks = client.export_label_box_v4_data_rows_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>>
            >>> with open('my-labelbox-rows.json', 'w') as f:
            >>>     json.dump(tasks, f)
        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_label_box_v4_data_rows_by_tag_id(tag.id)

    def export_filenames_by_tag_id(
        self,
        tag_id: str,
    ) -> str:
        """Fetches samples filenames within a certain tag by tag ID.

        More information:
        https://docs.lightly.ai/docs/filenames-and-readurls

        Args:
            tag_id:
                ID of the tag which should exported.

        Returns:
            A list of filenames of samples within a certain tag.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.export_filenames_by_tag_id("646b40d6c06aae1b91294a9e")
            'image-1.jpg\nimage-2.jpg\nimage-3.jpg'
        """
        filenames = "\n".join(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_basic_filenames,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
            )
        )
        return filenames

    def export_filenames_by_tag_name(
        self,
        tag_name: str,
    ) -> str:
        """Fetches samples filenames within a certain tag by tag name.

        More information:
        https://docs.lightly.ai/docs/filenames-and-readurls

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of filenames of samples within a certain tag.

        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> filenames = client.export_filenames_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>>
            >>> with open('filenames-of-initial-tag.txt', 'w') as f:
            >>>     f.write(filenames)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_filenames_by_tag_id(tag.id)

    def export_filenames_and_read_urls_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict[str, str]]:
        """Fetches filenames, read URLs, and datasource URLs from the given tag.

        More information:
        https://docs.lightly.ai/docs/filenames-and-readurls

        Args:
            tag_id:
                ID of the tag which should exported.

        Returns:
            A list of dictionaries with the keys "filename", "readUrl" and "datasourceUrl".
            An example:
            [
                {
                    "fileName": "sample1.jpg",
                    "readUrl": "s3://my_datasource/sample1.jpg?read_url_key=EAIFUIENDLFN",
                    "datasourceUrl": "s3://my_datasource/sample1.jpg",
                },
                {
                    "fileName": "sample2.jpg",
                    "readUrl": "s3://my_datasource/sample2.jpg?read_url_key=JSBFIEUHVSJ",
                    "datasourceUrl": "s3://my_datasource/sample2.jpg",
                },
            ]

        """
        filenames_string = "\n".join(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_basic_filenames,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
                file_name_format=FileNameFormat.NAME,
            )
        )
        read_urls_string = "\n".join(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_basic_filenames,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
                file_name_format=FileNameFormat.REDIRECTED_READ_URL,
            )
        )
        datasource_urls_string = "\n".join(
            utils.paginate_endpoint(
                self._tags_api.export_tag_to_basic_filenames,
                dataset_id=self.dataset_id,
                tag_id=tag_id,
                file_name_format=FileNameFormat.DATASOURCE_FULL,
            )
        )
        # The endpoint exportTagToBasicFilenames returns a plain string so we
        # have to split it by newlines in order to get the individual entries.
        # The order of the fileNames and readUrls and datasourceUrls is guaranteed to be the same
        # by the API so we can simply zip them.
        filenames = filenames_string.split("\n")
        read_urls = read_urls_string.split("\n")
        datasource_urls = datasource_urls_string.split("\n")
        return [
            {
                "fileName": filename,
                "readUrl": read_url,
                "datasourceUrl": datasource_url,
            }
            for filename, read_url, datasource_url in zip(
                filenames, read_urls, datasource_urls
            )
        ]

    def export_filenames_and_read_urls_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict[str, str]]:
        """Fetches filenames, read URLs, and datasource URLs from the given tag name.

        More information:
        https://docs.lightly.ai/docs/filenames-and-readurls

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of dictionaries with keys "filename", "readUrl" and "datasourceUrl".

        Examples:
            >>> # write json file which can be used to access the actual file contents.
            >>> mappings = client.export_filenames_and_read_urls_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>>
            >>> with open('my-samples.json', 'w') as f:
            >>>     json.dump(mappings, f)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_filenames_and_read_urls_by_tag_id(tag.id)
