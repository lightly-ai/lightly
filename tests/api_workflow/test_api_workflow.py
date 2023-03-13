import os
from unittest import mock

import numpy as np
from urllib3 import Timeout

import lightly
from tests.api_workflow.mocked_api_workflow_client import (
    MockedApiWorkflowClient,
    MockedApiWorkflowSetup,
)


class TestApiWorkflow(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        lightly.api.api_workflow_client.__version__ = lightly.__version__
        self.api_workflow_client = MockedApiWorkflowClient(token="token_xyz")

    @mock.patch.dict(os.environ, {"LIGHTLY_TOKEN": "token_xyz"})
    def test_init_with_env_token(self):
        MockedApiWorkflowClient()

    def test_error_if_init_without_token(self):
        # copy environment variables but remove LIGHTLY_TOKEN if it exists
        env_without_token = {
            k: v for k, v in os.environ.items() if k != "LIGHTLY_TOKEN"
        }
        with self.assertRaises(ValueError), mock.patch.dict(
            os.environ, env_without_token, clear=True
        ):
            MockedApiWorkflowClient()

    def test_error_if_version_is_incompatible(self):
        lightly.api.api_workflow_client.__version__ = "0.0.0"
        with self.assertWarns(UserWarning):
            MockedApiWorkflowClient(token="token_xyz")
        lightly.api.api_workflow_client.__version__ = lightly.__version__

    def test_dataset_id_nonexisting(self):
        self.api_workflow_client._datasets_api.reset()
        assert not hasattr(self.api_workflow_client, "_dataset_id")
        with self.assertWarns(UserWarning):
            dataset_id = self.api_workflow_client.dataset_id
        assert dataset_id == self.api_workflow_client._datasets_api.datasets[-1].id

    def test_dataset_id_existing(self):
        id = "random_dataset_id"
        self.api_workflow_client._dataset_id = id
        assert self.api_workflow_client.dataset_id == id

    def test_set_dataset_id_existing(self):
        self.api_workflow_client.dataset_id = "dataset_1_id"

    def test_set_dataset_id_missing(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.dataset_id = "nonexisting-id"

    def test_reorder_random(self):
        no_random_tries = 100
        for iter in range(no_random_tries):
            numbers_to_choose_from = list(range(100))
            numbers_all = list(np.random.choice(numbers_to_choose_from, 100))
            filenames_on_server = [f"img_{i}" for i in numbers_all]

            api_workflow_client = MockedApiWorkflowClient(
                token="token_xyz", dataset_id="dataset_id_xyz"
            )
            api_workflow_client._mappings_api.sample_names = filenames_on_server

            numbers_in_tag = np.copy(numbers_all)
            np.random.shuffle(numbers_in_tag)
            filenames_for_list = [f"img_{i}" for i in numbers_in_tag]

            list_ordered = api_workflow_client._order_list_by_filenames(
                filenames_for_list, list_to_order=numbers_in_tag
            )
            list_desired_order = [i for i in numbers_all if i in numbers_in_tag]
            assert list_ordered == list_desired_order

    def test_reorder_manual(self):
        filenames_on_server = ["a", "b", "c"]
        api_workflow_client = MockedApiWorkflowClient(
            token="token_xyz", dataset_id="dataset_id_xyz"
        )
        api_workflow_client._mappings_api.sample_names = filenames_on_server
        filenames_for_list = ["c", "a", "b"]
        list_to_order = ["cccc", "aaaa", "bbbb"]
        list_ordered = api_workflow_client._order_list_by_filenames(
            filenames_for_list, list_to_order=list_to_order
        )
        list_desired_order = ["aaaa", "bbbb", "cccc"]
        assert list_ordered == list_desired_order

    def test_reorder_wrong_lengths(self):
        filenames_on_server = ["a", "b", "c"]
        api_workflow_client = MockedApiWorkflowClient(
            token="token_xyz", dataset_id="dataset_id_xyz"
        )
        api_workflow_client._mappings_api.sample_names = filenames_on_server
        filenames_for_list = ["c", "a", "b"]
        list_to_order = ["cccc", "aaaa", "bbbb"]

        with self.subTest("filenames_for_list wrong length"):
            with self.assertRaises(ValueError):
                api_workflow_client._order_list_by_filenames(
                    filenames_for_list[:-1], list_to_order
                )

        with self.subTest("list_to_order wrong length"):
            with self.assertRaises(ValueError):
                api_workflow_client._order_list_by_filenames(
                    filenames_for_list, list_to_order[:-1]
                )

        with self.subTest("filenames_for_list and list_to_order wrong length"):
            with self.assertRaises(ValueError):
                api_workflow_client._order_list_by_filenames(
                    filenames_for_list[:-1], list_to_order[:-1]
                )
