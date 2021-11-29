from typing import *

from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client import TagData, \
    TagArithmeticsRequest, TagArithmeticsOperation, TagBitMaskResponse


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
            raise ValueError(f'Your tag_name is invalid: {tag_name}.')
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
                they need to be download newly, which is quite expensive.
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
                self._tags_api.perform_tag_arithmetics(
                    body=tag_arithmetics_request, dataset_id=self.dataset_id
                )
            bit_mask_data = bit_mask_response.bit_mask_data
        else:
            bit_mask_data = tag_data.bit_mask_data

        if not filenames_on_server:
            filenames_on_server = self.get_filenames()

        chosen_samples_ids = BitMask.from_hex(bit_mask_data).to_indices()
        filenames_tag = [filenames_on_server[i] for i in chosen_samples_ids]

        return filenames_tag
