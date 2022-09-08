from typing import List

from lightly.openapi_generated.swagger_client.models.shared_access_config_create_request import SharedAccessConfigCreateRequest
from lightly.openapi_generated.swagger_client.models.shared_access_config_data import SharedAccessConfigData
from lightly.openapi_generated.swagger_client.models.shared_access_type import SharedAccessType


class _CollaborationMixin:

    def share_dataset_only_with(self, dataset_id: str, user_emails: List[str]):
        """Shares dataset with a list of users

        This method overwrites the list of users that have had access to the dataset
        before. If you want to add someone new to the list make sure you get the
        list of users with access beforehand and add them as well.

        Args:
          dataset_id:
            Identifier of dataset
          user_emails:  
            List of email addresses of users to grant write permission

        Examples:
          >>> # share a dataset with a user
          >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
          >>> client.share_dataset_only_with(dataset_id="MY_DATASET_ID", user_emails=["user@something.com"])
          >>>
          >>> # share dataset with a user while keep sharing it with previous users
          >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
          >>> user_emails = client.get_shared_users(dataset_id="MY_DATASET_ID")
          >>> user_emails.append("additional_user2@something.com")
          >>> client.share_dataset_only_with(dataset_id="MY_DATASET_ID", user_emails=user_emails)
          >>>
          >>> # revoke access to all users
          >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
          >>> client.share_dataset_only_with(dataset_id="MY_DATASET_ID", user_emails=[])
        """
        body = SharedAccessConfigCreateRequest(access_type=SharedAccessType.WRITE, users=user_emails)
        self._collaboration_api.create_or_update_shared_access_config_by_dataset_id(body=body, dataset_id=dataset_id)


    def get_shared_users(self, dataset_id: str) -> List[str]:
      """Get list of users that have access to the dataset
      
      Args:
        dataset_id:
          Identifier of dataset
      
      Returns:
        List of email addresses of users that have write access to the dataset

      Examples:
          >>> client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")
          >>> client.get_shared_users(dataset_id="MY_DATASET_ID")
          >>> ["user@something.com"]
      """

      access_configs: List[SharedAccessConfigData] = self._collaboration_api.get_shared_access_configs_by_dataset_id(dataset_id=dataset_id)
      user_emails = []

      # iterate through configs and find first WRITE config
      # we use the same hard rule in the frontend to communicate with the API
      # as we currently only support WRITE access
      for access_config in access_configs:
        if access_config.access_type == SharedAccessType.WRITE:
          user_emails.extend(access_config.users)
          break

      return user_emails
