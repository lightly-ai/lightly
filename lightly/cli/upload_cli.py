# -*- coding: utf-8 -*-
"""**Lightly Upload:** Upload images to the Lightly platform.

This module contains the entrypoint for the **lightly-upload**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import warnings

import hydra

import torchvision
from torch.utils.hipify.hipify_python import bcolors

from lightly.cli._helpers import fix_input_path, print_as_warning

from lightly.api.utils import getenv
from lightly.api.api_workflow_client import ApiWorkflowClient
from lightly.data import LightlyDataset


def _upload_cli(cfg, is_cli_call=True):
    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)

    path_to_embeddings = cfg['embeddings']
    if path_to_embeddings and is_cli_call:
        path_to_embeddings = fix_input_path(path_to_embeddings)

    dataset_id = cfg['dataset_id']
    token = cfg['token']
    new_dataset_name = cfg['new_dataset_name']

    cli_api_args_wrong = False
    if not token:
        print_as_warning('Please specify your access token.')
        cli_api_args_wrong = True

    dataset_id_ok = dataset_id and len(dataset_id) > 0
    new_dataset_name_ok = new_dataset_name and len(new_dataset_name) > 0
    if new_dataset_name_ok and not dataset_id_ok:
        api_workflow_client = ApiWorkflowClient(token=token)
        api_workflow_client.create_dataset(dataset_name=new_dataset_name)
    elif dataset_id_ok and not new_dataset_name_ok:
        api_workflow_client = ApiWorkflowClient(token=token, dataset_id=dataset_id)
    else:
        print_as_warning('Please specify either the dataset_id of an existing dataset or a new_dataset_name.')
        cli_api_args_wrong = True

    if cli_api_args_wrong:
        print_as_warning('For help, try: lightly-upload --help')
        return

    size = cfg['resize']
    if not isinstance(size, int):
        size = tuple(size)
    transform = None
    if isinstance(size, tuple) or size > 0:
        transform = torchvision.transforms.Resize(size)

    if input_dir:
        mode = cfg['upload']
        dataset = LightlyDataset(input_dir=input_dir, transform=transform)
        api_workflow_client.upload_dataset(
            input=dataset, mode=mode, max_workers=cfg['loader']['num_workers']
        )

    if path_to_embeddings:
        name = cfg['embedding_name']
        api_workflow_client.upload_embeddings(
            path_to_embeddings_csv=path_to_embeddings, name=name
        )
    if new_dataset_name_ok:
        print(f'The dataset_id of the newly created dataset is '
              f'{bcolors.OKBLUE}{api_workflow_client.dataset_id}{bcolors.ENDC}')


@hydra.main(config_path='config', config_name='config')
def upload_cli(cfg):
    """Upload images/embeddings from the command-line to the Lightly platform.

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file 
            (.config.config.yaml).
    
    Command-Line Args:
        input_dir:
            Path to the input directory where images are stored.
        embeddings:
            Path to the csv file storing the embeddings generated by
            lightly.
        token:
            User access token to the Lightly platform. If needs to be
            specified to upload the images and embeddings to the platform.
        dataset_id:
            Identifier of the dataset on the Lightly platform.
            Either the dataset_id or the new_dataset_name need to be
            specified.
        new_dataset_name:
            The name of the new dataset to create on the Lightly platform.
            Either the dataset_id or the new_dataset_name need to be
            specified.
        upload:
            String to determine whether to upload the full images, 
            thumbnails only, or metadata only.

            Must be one of ['full', 'thumbnails', 'metadata']
        embedding_name:
            Assign the embedding a name in order to identify it on the 
            Lightly platform.
        resize:
            Desired size of the uploaded images. If negative, default size is used.
            If size is a sequence like (h, w), output size will be matched to 
            this. If size is an int, smaller edge of the image will be matched 
            to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size).

    Examples:
        >>> # create a new dataset on the Lightly platform and upload thumbnails to it
        >>> lightly-upload input_dir=data/ token='123' new_dataset_name='new_dataset_name_xyz'
        >>>
        >>> # upload thumbnails to the Lightly platform to an existing dataset
        >>> lightly-upload input_dir=data/ token='123' dataset_id='XYZ'
        >>> 
        >>> # create a new dataset on the Lightly platform and upload full images to it
        >>> lightly-upload input_dir=data/ token='123' new_dataset_name='new_dataset_name_xyz' upload='full'
        >>>
        >>> # upload metadata to the Lightly platform
        >>> lightly-upload input_dir=data/ token='123' dataset_id='XYZ' upload='metadata'
        >>>
        >>> # upload embeddings to the Lightly platform (must have uploaded images beforehand)
        >>> lightly-upload embeddings=embeddings.csv token='123' dataset_id='XYZ'
        >>>
        >>> # upload both, images and embeddings in a single command
        >>> lightly-upload input_dir=data/ embeddings=embeddings.csv upload='full' \\
        >>>     token='123' dataset_id='XYZ'
        >>> # create a new dataset on the Lightly platform and upload both, images and embeddings
        >>> lightly-upload input_dir=data/ embeddings=embeddings.csv upload='full' \\
        >>>     token='123' new_dataset_name='new_dataset_name_xyz'

    """
    _upload_cli(cfg)


def entry():
    upload_cli()
