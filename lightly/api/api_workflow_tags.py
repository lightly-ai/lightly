from typing import *

from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client.models import (
    TagArithmeticsOperation,
    TagArithmeticsRequest,
    TagBitMaskResponse,
    TagCreateRequest,
    TagData,
)


class TagDoesNotExistError(ValueError):
    pass


class _TagsMixin:
    def get_all_tags(self) -> List[TagData]:
        """Gets all tags in the Lightly Platform from the current dataset.

        Returns:
            A list of tags.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.get_all_tags()
            [{'created_at': 1684750550014,
             'dataset_id': '646b40a18355e2f54c6d2200',
             'id': '646b40d6c06aae1b91294a9e',
             'last_modified_at': 1684750550014,
             'name': 'cool-tag',
             'preselected_tag_id': None,
             ...}]
        """
        return self._tags_api.get_tags_by_dataset_id(self.dataset_id)

    def get_tag_by_id(self, tag_id: str) -> TagData:
        """Gets a tag from the current dataset by tag ID.

        Args:
            tag_id:
                ID of the requested tag.

        Returns:
            Tag data for the requested tag.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.get_tag_by_id("646b40d6c06aae1b91294a9e")
            {'created_at': 1684750550014,
             'dataset_id': '646b40a18355e2f54c6d2200',
             'id': '646b40d6c06aae1b91294a9e',
             'last_modified_at': 1684750550014,
             'name': 'cool-tag',
             'preselected_tag_id': None,
             ...}
        """
        tag_data = self._tags_api.get_tag_by_tag_id(
            dataset_id=self.dataset_id, tag_id=tag_id
        )
        return tag_data

    def get_tag_by_name(self, tag_name: str) -> TagData:
        """Gets a tag from the current dataset by tag name.

        Args:
            tag_name:
                Name of the requested tag.

        Returns:
            Tag data for the requested tag.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> client.get_tag_by_name("cool-tag")
            {'created_at': 1684750550014,
             'dataset_id': '646b40a18355e2f54c6d2200',
             'id': '646b40d6c06aae1b91294a9e',
             'last_modified_at': 1684750550014,
             'name': 'cool-tag',
             'preselected_tag_id': None,
             ...}
        """
        tag_name_id_dict = {tag.name: tag.id for tag in self.get_all_tags()}
        tag_id = tag_name_id_dict.get(tag_name, None)
        if tag_id is None:
            raise TagDoesNotExistError(f"Your tag_name does not exist: {tag_name}.")
        return self.get_tag_by_id(tag_id)

    def get_filenames_in_tag(
        self,
        tag_data: TagData,
        filenames_on_server: List[str] = None,
        exclude_parent_tag: bool = False,
    ) -> List[str]:
        """Gets the filenames of samples under a tag.

        Args:
            tag_data:
                Information about the tag.
            filenames_on_server:
                List of all filenames on the server. If they are not given,
                they need to be downloaded, which is a time-consuming operation.
            exclude_parent_tag:
                Excludes the parent tag in the returned filenames.

        Returns:
            Filenames of all samples under the tag.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> tag = client.get_tag_by_name("cool-tag")
            >>> client.get_filenames_in_tag(tag_data=tag)
            ['image-1.png', 'image-2.png']

        :meta private:  # Skip docstring generation
        """

        if exclude_parent_tag:
            parent_tag_id = tag_data.prev_tag_id
            tag_arithmetics_request = TagArithmeticsRequest(
                tag_id1=tag_data.id,
                tag_id2=parent_tag_id,
                operation=TagArithmeticsOperation.DIFFERENCE,
            )
            bit_mask_response: TagBitMaskResponse = (
                self._tags_api.perform_tag_arithmetics_bitmask(
                    tag_arithmetics_request=tag_arithmetics_request,
                    dataset_id=self.dataset_id,
                )
            )
            bit_mask_data = bit_mask_response.bit_mask_data
        else:
            bit_mask_data = tag_data.bit_mask_data

        if not filenames_on_server:
            filenames_on_server = self.get_filenames()

        filenames_tag = BitMask.from_hex(bit_mask_data).masked_select_from_list(
            filenames_on_server
        )

        return filenames_tag

    def create_tag_from_filenames(
        self, fnames_new_tag: List[str], new_tag_name: str, parent_tag_id: str = None
    ) -> TagData:
        """Creates a new tag from a list of filenames.

        Args:
            fnames_new_tag:
                A list of filenames to be included in the new tag.
            new_tag_name:
                The name of the new tag.
            parent_tag_id:
                The tag defining where to sample from, default: None resolves to the initial-tag.

        Returns:
            The newly created tag.

        Raises:
            RuntimeError:
                When a tag with the desired tag name already exists.
                When `initial-tag` does not exist.
                When any of the given files does not exist.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> filenames = ['image-1.png', 'image-2.png']
            >>> client.create_tag_from_filenames(fnames_new_tag=filenames, new_tag_name='new-tag')
            {'id': '6470c4c1060894655c5a8ed5'}
        """

        # make sure the tag name does not exist yet
        tags = self.get_all_tags()
        if new_tag_name in [tag.name for tag in tags]:
            raise RuntimeError(
                f"There already exists a tag with tag_name {new_tag_name}."
            )
        if len(tags) == 0:
            raise RuntimeError("There exists no initial-tag for this dataset.")

        # fallback to initial tag if no parent tag is provided
        if parent_tag_id is None:
            parent_tag_id = next(tag.id for tag in tags if tag.name == "initial-tag")

        # get list of filenames from tag
        fnames_server = self.get_filenames()
        tot_size = len(fnames_server)

        # create new bitmask for the new tag
        bitmask = BitMask(0)
        fnames_new_tag = set(fnames_new_tag)
        for i, fname in enumerate(fnames_server):
            if fname in fnames_new_tag:
                bitmask.set_kth_bit(i)

        # quick sanity check
        num_selected_samples = len(bitmask.to_indices())
        if num_selected_samples != len(fnames_new_tag):
            raise RuntimeError(
                "An error occurred when creating the new subset! "
                f"Out of the {len(fnames_new_tag)} filenames you provided "
                f"to create a new tag, only {num_selected_samples} have been "
                "found on the server. "
                "Make sure you use the correct filenames. "
                f"Valid filename example from the dataset: {fnames_server[0]}"
            )

        # create new tag
        tag_data_dict = {
            "name": new_tag_name,
            "prevTagId": parent_tag_id,
            "bitMaskData": bitmask.to_hex(),
            "totSize": tot_size,
            "creator": self._creator,
        }

        new_tag = self._tags_api.create_tag_by_dataset_id(
            tag_create_request=TagCreateRequest.from_dict(tag_data_dict),
            dataset_id=self.dataset_id,
        )

        return new_tag

    def delete_tag_by_id(self, tag_id: str) -> None:
        """Deletes a tag from the current dataset.

        Args:
            tag_id:
                The id of the tag to be deleted.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> filenames = ['image-1.png', 'image-2.png']
            >>> tag_id = client.create_tag_from_filenames(fnames_new_tag=filenames, new_tag_name='new-tag')["id"]
            >>> client.delete_tag_by_id(tag_id=tag_id)
        """
        self._tags_api.delete_tag_by_tag_id(dataset_id=self.dataset_id, tag_id=tag_id)

    def delete_tag_by_name(self, tag_name: str) -> None:
        """Deletes a tag from the current dataset.

        Args:
            tag_name:
                The name of the tag to be deleted.

        Examples:
            >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
            >>>
            >>> # Already created some Lightly Worker runs with this dataset
            >>> client.set_dataset_id_by_name("my-dataset")
            >>> filenames = ['image-1.png', 'image-2.png']
            >>> client.create_tag_from_filenames(fnames_new_tag=filenames, new_tag_name='new-tag')
            >>> client.delete_tag_by_name(tag_name="new-tag")
        """
        tag_data = self.get_tag_by_name(tag_name=tag_name)
        self.delete_tag_by_id(tag_data.id)
