import os
import re
import sys
import tempfile
import random

import torchvision
import yaml
from hydra.experimental import compose, initialize

import lightly
from lightly.active_learning.utils import BoundingBox
from lightly.data import LightlyDataset
from lightly.utils.cropping.crop_image_by_bounding_boxes import crop_dataset_by_bounding_boxes_and_save
from lightly.utils.cropping.read_yolo_label_file import read_yolo_label_file
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestCLICrop(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.upload_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def setUp(self):
        MockedApiWorkflowSetup.setUp(self)
        self.create_fake_dataset()
        self.create_fake_yolo_labels()
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config", overrides=[
                f"input_dir={self.folder_path}",
                f"label_dir={self.folder_path_labels}",
                f"output_dir={tempfile.mkdtemp()}",
                f"label_names_file={self.label_names_file}"
            ])

    def create_fake_dataset(self):
        n_data = len(self.api_workflow_client.filenames_on_server)
        self.dataset = torchvision.datasets.FakeData(size=n_data, image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)

    def create_fake_yolo_labels(self, no_classes: int = 10, objects_per_image: int = 13):
        random.seed(42)

        n_data = len(self.api_workflow_client.filenames_on_server)

        self.folder_path_labels = tempfile.mkdtemp()
        label_names = [f'img_{i}.txt' for i in range(n_data)]
        self.label_names = label_names
        for filename_label in label_names:
            path = os.path.join(self.folder_path_labels, filename_label)
            with open(path, 'a') as the_file:
                for i in range(objects_per_image):
                    class_id = random.randint(0, no_classes - 1)
                    x = random.uniform(0.1, 0.9)
                    y = random.uniform(0.1, 0.9)
                    w = random.uniform(0.1, 1.0)
                    h = random.uniform(0.1, 1.0)
                    line = f"{class_id} {x} {y} {w} {h}\n"
                    the_file.write(line)
        yaml_dict = {"names": [f"class{i}" for i in range(no_classes)]}
        self.label_names_file = tempfile.mktemp('.yaml', 'data', dir=self.folder_path_labels)
        with open(self.label_names_file, 'w') as file:
            yaml.dump(yaml_dict, file)

    def parse_cli_string(self, cli_words: str):
        cli_words = cli_words.replace("lightly-crop ", "")
        cli_words = re.split("=| ", cli_words)
        assert len(cli_words) % 2 == 0
        dict_keys = cli_words[0::2]
        dict_values = cli_words[1::2]
        for key, value in zip(dict_keys, dict_values):
            value = value.strip('\"')
            value = value.strip('\'')
            self.cfg[key] = value

    def test_parse_cli_string(self):
        cli_string = "lightly-crop label_dir=/blub"
        self.parse_cli_string(cli_string)
        self.assertEqual(self.cfg['label_dir'], '/blub')

    def test_read_yolo(self):
        for f in os.listdir(self.cfg.label_dir):
            if f.endswith('.txt'):
                filepath = os.path.join(self.cfg.label_dir, f)
                read_yolo_label_file(filepath, 0.1)

    def test_crop_dataset_by_bounding_boxes_and_save(self):
        dataset = LightlyDataset(self.cfg.input_dir)
        output_dir = self.cfg.output_dir
        no_files = len(dataset.get_filenames())
        bounding_boxes_list_list = [[BoundingBox(0, 0, 1, 1)]] * no_files
        class_indices_list_list = [[1]] * no_files
        class_names = ["class_0", "class_1"]
        with self.subTest("all_correct"):
            crop_dataset_by_bounding_boxes_and_save(dataset,
                                                    output_dir,
                                                    bounding_boxes_list_list,
                                                    class_indices_list_list,
                                                    class_names)
        with self.subTest("wrong length of bounding_boxes_list_list"):
            with self.assertRaises(ValueError):
                crop_dataset_by_bounding_boxes_and_save(dataset,
                                                        output_dir,
                                                        bounding_boxes_list_list[:-1],
                                                        class_indices_list_list,
                                                        class_names)
        with self.subTest("wrong internal length of class_indices_list_list"):
            with self.assertRaises(ValueError):
                class_indices_list_list[0] *= 2
                crop_dataset_by_bounding_boxes_and_save(dataset,
                                                        output_dir,
                                                        bounding_boxes_list_list[:-1],
                                                        class_indices_list_list,
                                                        class_names)



    def test_crop_with_class_names(self):
        cli_string = "lightly-crop crop_padding=0.1"
        self.parse_cli_string(cli_string)
        lightly.cli.crop_cli(self.cfg)

    def test_crop_without_class_names(self):
        cli_string = "lightly-crop crop_padding=0.1"
        self.parse_cli_string(cli_string)
        self.cfg['label_names_file'] = ''
        lightly.cli.crop_cli(self.cfg)
