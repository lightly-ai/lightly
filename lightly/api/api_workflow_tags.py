from typing import *

from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client import TagData, \
    TagArithmeticsRequest, TagArithmeticsOperation, TagBitMaskResponse


class _TagsMixin:

    def _get_all_tags(self) -> List[TagData]:
        """ Gets all tags on the server

        Returns:

        """
        return self._tags_api.get_tags_by_dataset_id(self.dataset_id)

    def get_tag_and_filenames(
            self,
            tag_name: str = None,
            tag_id: str = None,
            filenames_on_server: List[str] = None,
            exclude_parent_tag: bool = False
    ) -> Tuple[TagData, List[str]]:
        """ Gets the TagData of a tag and the filenames in it

        The tag_name or the tag_id must be passed.
        The tag_name overwrites the tag_id.
        Args:
            tag_name:
                The name of the tag.
            tag_id:
                The id of the tag.
            filenames_on_server:
                List of all filenames on the server. If they are not given,
                they need to be download newly, which is quite expensive.
            exclude_parent_tag:
                Excludes the parent tag in the returned filenames.

        Returns:
            tag_data:
                The tag_data, raw from the API
            filenames_tag:
                The filenames of all samples in the tag.

        """
        tag_name_id_dict = {tag.name: tag.id for tag in self._get_all_tags()}
        if tag_name:

            tag_id = tag_name_id_dict.get(tag_name, None)
            if tag_id is None:
                raise ValueError(f'Your tag_name is invalid: {tag_name}.')
        elif tag_id is None or tag_id not in tag_name_id_dict.values():
            raise ValueError(f'Your tag_id is invalid: {tag_id}')

        tag_data = self._tags_api.get_tag_by_tag_id(self.dataset_id, tag_id)

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
            filenames_on_server = self.download_filenames_from_server()

        chosen_samples_ids = BitMask.from_hex(bit_mask_data).to_indices()
        filenames_tag = [filenames_on_server[i] for i in chosen_samples_ids]

        return tag_data, filenames_tag