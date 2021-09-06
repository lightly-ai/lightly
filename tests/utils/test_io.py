import sys
import tempfile

from lightly.utils import save_custom_metadata
from lightly.utils.io import check_filenames
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestCLICrop(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.upload_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def test_save_metadata(self):
        metadata = [("filename.jpg", {"random_metadata": 42})]
        metadata_filepath = tempfile.mktemp('.json', 'metadata')
        save_custom_metadata(metadata_filepath, metadata)

    def test_valid_filenames(self):
        valid = 'img.png'
        non_valid = 'img,1.png'
        filenames_list = [
            ([valid], True),
            ([valid, valid], True),
            ([non_valid], False),
            ([valid, non_valid], False),
        ]
        for filenames, valid in filenames_list:
            with self.subTest(msg=f"filenames:{filenames}"):
                if valid:
                    check_filenames(filenames)
                else:
                    with self.assertRaises(ValueError):
                        check_filenames(filenames)
