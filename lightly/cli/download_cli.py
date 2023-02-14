# -*- coding: utf-8 -*-
"""**Lightly Download:** Download images from the Lightly platform.

This module contains the entrypoint for the **lightly-download**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import shutil
import warnings

import hydra
from torch.utils.hipify.hipify_python import bcolors
from tqdm import tqdm

import lightly.data as data
from lightly.cli._helpers import fix_input_path
from lightly.cli._helpers import print_as_warning
from lightly.cli._helpers import fix_hydra_arguments
from lightly.cli._helpers import cpu_count

from lightly.api.utils import getenv
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client import TagData, TagArithmeticsRequest, TagArithmeticsOperation, TagBitMaskResponse, DatasetCreator


def _download_cli(cfg, is_cli_call=True):

    tag_name = str(cfg['tag_name'])
    dataset_id = str(cfg['dataset_id'])
    token = str(cfg['token'])

    if not tag_name or not token or not dataset_id:
        print_as_warning('Please specify all of the parameters tag_name, token and dataset_id')
        print_as_warning('For help, try: lightly-download --help')
        return

    # set the number of workers if unset
    if cfg['loader']['num_workers'] < 0:
        # set the number of workers to the number of CPUs available,
        # but minimum of 8
        num_workers = max(8, cpu_count())
        num_workers = min(32, num_workers)
        cfg['loader']['num_workers'] = num_workers

    api_workflow_client = ApiWorkflowClient(
        token=token, dataset_id=dataset_id, dataset_creator=DatasetCreator.USER_PIP_LIGHTLY_MAGIC
    )

    # get tag id
    tag_data = api_workflow_client.get_tag_by_name(tag_name)
    filenames_tag = api_workflow_client.get_filenames_in_tag(
        tag_data,
        exclude_parent_tag=cfg['exclude_parent_tag'],
    )

    # store sample names in a .txt file
    filename = tag_name + '.txt'
    with open(filename, 'w') as f:
        for item in filenames_tag:
            f.write("%s\n" % item)

    filepath = os.path.join(os.getcwd(), filename)
    msg = f'The list of samples in tag {cfg["tag_name"]} is stored at: {bcolors.OKBLUE}{filepath}{bcolors.ENDC}'
    print(msg, flush=True)

    if not cfg['input_dir'] and cfg['output_dir']:
        # download full images from api
        output_dir = fix_input_path(cfg['output_dir'])
        api_workflow_client.download_dataset(
            output_dir,
            tag_name=tag_name,
            max_workers=cfg['loader']['num_workers']
        )

    elif cfg['input_dir'] and cfg['output_dir']:
        input_dir = fix_input_path(cfg['input_dir'])
        output_dir = fix_input_path(cfg['output_dir'])
        print(f'Copying files from {input_dir} to {bcolors.OKBLUE}{output_dir}{bcolors.ENDC}.')

        # create a dataset from the input directory
        dataset = data.LightlyDataset(input_dir=input_dir)

        # dump the dataset in the output directory
        dataset.dump(output_dir, filenames_tag)


@hydra.main(**fix_hydra_arguments(config_path = 'config', config_name = 'config'))
def download_cli(cfg):
    """Download images from the Lightly platform.

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file 
            (.config.config.yaml).
    
    Command-Line Args:
        tag_name:
            Download all images from the requested tag. Use initial-tag
            to get all images from the dataset.
        token:
            User access token to the Lightly platform. If dataset_id
            and token are specified, the images and embeddings are 
            uploaded to the platform.
        dataset_id:
            Identifier of the dataset on the Lightly platform. If 
            dataset_id and token are specified, the images and 
            embeddings are uploaded to the platform.
        input_dir:
            If input_dir and output_dir are specified, lightly will copy
            all images belonging to the tag from the input_dir to the 
            output_dir.
        output_dir:
            If input_dir and output_dir are specified, lightly will copy
            all images belonging to the tag from the input_dir to the 
            output_dir.

    Examples:
        >>> # download list of all files in the dataset from the Lightly platform
        >>> lightly-download token='123' dataset_id='XYZ'
        >>> 
        >>> # download list of all files in tag 'my-tag' from the Lightly platform
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag'
        >>>
        >>> # download all images in tag 'my-tag' from the Lightly platform
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag' output_dir='my_data/'
        >>>
        >>> # copy all files in 'my-tag' to a new directory
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag' input_dir='data/' output_dir='my_data/'


    """
    _download_cli(cfg)


def entry():
    download_cli()
