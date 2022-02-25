# -*- coding: utf-8 -*-
"""**Lightly Upload:** Upload images to the Lightly platform.

This module contains the entrypoint for the **lightly-upload**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import csv
import json
import os
from datetime import datetime

import hydra

import torchvision
from torch.utils.hipify.hipify_python import bcolors

from lightly.cli._helpers import fix_input_path, print_as_warning, cpu_count

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

    if dataset_id:
        if new_dataset_name:
            print_as_warning(
                'Please specify either the dataset_id of an existing dataset '
                'or a new_dataset_name, but not both.'
            )
            cli_api_args_wrong = True
        else:
            api_workflow_client = \
                ApiWorkflowClient(token=token, dataset_id=dataset_id)
    else:
        if new_dataset_name:
            api_workflow_client = ApiWorkflowClient(token=token)
            api_workflow_client.create_dataset(dataset_name=new_dataset_name)
        else:
            print_as_warning(
                'Please specify either the dataset_id of an existing dataset '
                'or a new_dataset_name.')
            cli_api_args_wrong = True
    # delete the dataset_id as it might be an empty string
    # Use api_workflow_client.dataset_id instead
    del dataset_id

    if cli_api_args_wrong:
        print_as_warning('For help, try: lightly-upload --help')
        return

    # potentially load custom metadata
    custom_metadata = None
    if cfg['custom_metadata']:
        path_to_custom_metadata = fix_input_path(cfg['custom_metadata'])
        print(
            'Loading custom metadata from '
            f'{bcolors.OKBLUE}{path_to_custom_metadata}{bcolors.ENDC}'
        )
        with open(path_to_custom_metadata, 'r') as f:
            custom_metadata = json.load(f)

    # set the number of workers if unset
    if cfg['loader']['num_workers'] < 0:
        # set the number of workers to the number of CPUs available,
        # but minimum of 8
        num_workers = max(8, cpu_count())
        num_workers = min(32, num_workers)
        cfg['loader']['num_workers'] = num_workers

    size = cfg['resize']
    if not isinstance(size, int):
        size = tuple(size)
    transform = None
    if isinstance(size, tuple) or size > 0:
        transform = torchvision.transforms.Resize(size)

    if input_dir:
        if len(api_workflow_client.get_all_tags()) > 0:
            if not cfg.append:
                print_as_warning(
                    'The dataset you specified already has samples. '
                    'If you want to add additional samples, you need to specify '
                    'append=True as CLI argument.'
                )
                return

        mode = cfg['upload']
        dataset = LightlyDataset(input_dir=input_dir, transform=transform)
        api_workflow_client.upload_dataset(
            input=dataset,
            mode=mode,
            max_workers=cfg['loader']['num_workers'],
            custom_metadata=custom_metadata,
        )
        print('Finished the upload of the dataset.')

    if path_to_embeddings:
        name = cfg['embedding_name']
        print('Starting upload of embeddings.')
        api_workflow_client.upload_embeddings(
            path_to_embeddings_csv=path_to_embeddings, name=name
        )
        print('Finished upload of embeddings.')

    if custom_metadata is not None and not input_dir:
        # upload custom metadata separately
        api_workflow_client.upload_custom_metadata(
            custom_metadata,
            verbose=True,
            max_workers=cfg['loader']['num_workers'],
        )

    if new_dataset_name:
        print(f'The dataset_id of the newly created dataset is '
              f'{bcolors.OKBLUE}{api_workflow_client.dataset_id}{bcolors.ENDC}')

    os.environ[
        cfg['environment_variable_names']['lightly_last_dataset_id']
    ] = api_workflow_client.dataset_id


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
            Desired size of the uploaded images. If negative, default size is
            used. If size is a sequence like (h, w), output size will be matched
            to this. If size is an int, smaller edge of the image will be
            matched to this number. i.e, if height > width, then image will be
            rescaled to (size * height / width, size).
        custom_metadata:
            Path to a .json file containing custom metadata. The file must be in
            the COCO annotations (although annotations can be empty) format and
            contain an additional field `metadata` storing a list of metadata
            entries. The metadata entries are matched with the images via
            `image_id`.

    Examples:
        >>> # create a new dataset on the Lightly platform and upload full images to it
        >>> lightly-upload input_dir=data/ token='123' new_dataset_name='new_dataset_name_xyz'
        >>>
        >>> # upload full images to the Lightly platform to an existing dataset
        >>> lightly-upload input_dir=data/ token='123' dataset_id='XYZ'
        >>>
        >>> # create a new dataset on the Lightly platform and upload thumbnails to it
        >>> lightly-upload input_dir=data/ token='123' new_dataset_name='new_dataset_name_xyz' upload='thumbnails'
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
        >>>
        >>> # create a new dataset on the Lightly platform and upload both, images and embeddings
        >>> lightly-upload input_dir=data/ embeddings=embeddings.csv upload='full' \\
        >>>     token='123' new_dataset_name='new_dataset_name_xyz'
        >>>
        >>> # upload a dataset with custom metadata
        >>> lightly-upload input_dir=data/ token='123' dataset_id='XYZ' custom_metadata=custom_metadata.json
        >>>
        >>> # upload custom metadata to an existing dataset
        >>> lightly-upload token='123' dataset_id='XYZ' custom_metadata=custom_metadata.json

    """
    _upload_cli(cfg)


def entry():
    upload_cli()
