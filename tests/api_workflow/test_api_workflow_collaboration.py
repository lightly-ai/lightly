from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestApiWorkflowDatasets(MockedApiWorkflowSetup):

    def setUp(self) -> None:
        self.api_workflow_client = MockedApiWorkflowClient(token="token_xyz")

    def test_share_empty_dataset(self):
        self.api_workflow_client.share_dataset_only_with(dataset_id="some-dataset-id", user_emails=[])

    def test_share_dataset(self):
        self.api_workflow_client.share_dataset_only_with(dataset_id="some-dataset-id", user_emails=["someone@something.com"])

    def test_get_shared_users(self):
        user_emails = self.api_workflow_client.get_shared_users(dataset_id="some-dataset-id")
        assert user_emails == ["user1@gmail.com", "user2@something.com"]
