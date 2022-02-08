from typing import *

from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client import TagData, \
    TagArithmeticsRequest, TagArithmeticsOperation, TagBitMaskResponse


class TagDoesNotExistError(ValueError):
    pass

class _TagsMixin:

    def get_all_tags(self) -> List[TagData]:
        """ Gets all tags on the server

        Returns:
            one TagData entry for each tag on the server

        """
        return self._tags_api.get_tags_by_dataset_id(self.dataset_id)

    def get_tag_by_id(self, tag_id: str) -> TagData:
        tag_data = self._tags_api.get_tag_by_tag_id(self.dataset_id, tag_id)
        return tag_data

    def get_tag_by_name(self, tag_name: str) -> TagData:
        tag_name_id_dict = {tag.name: tag.id for tag in self.get_all_tags()}
        tag_id = tag_name_id_dict.get(tag_name, None)
        if tag_id is None:
            raise TagDoesNotExistError(f'Your tag_name does not exist: {tag_name}.')
        return self.get_tag_by_id(tag_id)

    def get_filenames_in_tag(
            self,
            tag_data: TagData,
            filenames_on_server: List[str] = None,
            exclude_parent_tag: bool = False,
    ) -> List[str]:
        """ Gets the filenames of a tag

        Args:
            tag_data:
                The data of the tag.
            filenames_on_server:
                List of all filenames on the server. If they are not given,
                they need to be downloaded, which is quite expensive.
            exclude_parent_tag:
                Excludes the parent tag in the returned filenames.

        Returns:
            filenames_tag:
                The filenames of all samples in the tag.

        """

        if exclude_parent_tag:
            parent_tag_id = tag_data.prev_tag_id
            tag_arithmetics_request = TagArithmeticsRequest(
                tag_id1=tag_data.id, tag_id2=parent_tag_id,
                operation=TagArithmeticsOperation.DIFFERENCE)
            bit_mask_response: TagBitMaskResponse = \
                self._tags_api.perform_tag_arithmetics_bitmask(
                    body=tag_arithmetics_request, dataset_id=self.dataset_id
                )
            bit_mask_data = bit_mask_response.bit_mask_data
        else:
            bit_mask_data = tag_data.bit_mask_data

        if not filenames_on_server:
            filenames_on_server = self.get_filenames()

        filenames_tag = BitMask.from_hex(bit_mask_data).\
            masked_select_from_list(filenames_on_server)

        return filenames_tag

    def create_tag_from_filenames(
        self,
        fnames_new_tag: List[str],
        new_tag_name: str,
        parent_tag_id: str = None
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
            RuntimeError
        """
        
        # make sure the tag name does not exist yet
        tags = self.get_all_tags()
        if new_tag_name in [tag.name for tag in tags]:
            raise RuntimeError(f'There already exists a tag with tag_name {new_tag_name}.')
        if len(tags) == 0:
            raise RuntimeError('There exists no initial-tag for this dataset.')

        # fallback to initial tag if no parent tag is provided
        if parent_tag_id is None:
            parent_tag_id = tags[-1].id

        tot_size = tags[-1].tot_size

        # get list of filenames from tag
        fnames_server = self.get_filenames()

        # create new bitmask 
        bitmask = BitMask(tot_size)
        for i, fname_server in enumerate(fnames_server):
            bitmask.unset_kth_bit(i)
            if fname_server in fnames_new_tag:
                bitmask.set_kth_bit(i)

        
        # quick sanity check
        num_selected_samples = len(bitmask.to_indices())
        if num_selected_samples != len(fnames_new_tag):
            raise RuntimeError(
                f'An error occured when creating the new subset! '
                f'Found {num_selected_samples} samples newly selected '
                f'instead of {len(fnames_new_tag)}. '
                f'Make sure you use the correct filenames. '
                f'Valid filename example from the dataset: {fnames_server[0]}'
                )

        # create new tag
        tag_data_dict = {
            'name': new_tag_name, 
            'prevTagId': parent_tag_id, 
            'bitMaskData': bitmask.to_hex(), 
            'totSize': tot_size
        }

        new_tag = self._tags_api.create_tag_by_dataset_id(
            tag_data_dict,
            self.dataset_id
        )

        return new_tag
