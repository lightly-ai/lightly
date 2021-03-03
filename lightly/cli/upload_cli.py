# -*- coding: utf-8 -*-
"""**Lightly Upload:** Upload images to the Lightly platform.

This module contains the entrypoint for the **lightly-upload**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import hydra

from lightly.cli._helpers import fix_input_path

from lightly.api.utils import getenv
from lightly.api.api_workflow_client import ApiWorkflowClient


def _upload_cli(cfg, is_cli_call=True):

    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)

    path_to_embeddings = cfg['embeddings']
    if path_to_embeddings and is_cli_call:
        path_to_embeddings = fix_input_path(path_to_embeddings)

    dataset_id = cfg['dataset_id']
    token = cfg['token']

    size = cfg['resize']
    if not isinstance(size, int):
        size = tuple(size)

    if not token or not dataset_id:
        print('Please specify your access token and dataset id.')
        print('For help, try: lightly-upload --help')
        return

    api_workflow_client = ApiWorkflowClient(
        token=token, dataset_id=dataset_id
    )

    if input_dir:
        mode = cfg['upload']
        api_workflow_client.upload_dataset(
            input=input_dir, mode=mode, size=size
        )

    if path_to_embeddings:
        name = cfg['embedding_name']
        api_workflow_client.upload_embeddings(
            path_to_embeddings_csv=path_to_embeddings, name=name
        )


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
            User access token to the Lightly platform. If dataset_id
            and token are specified, the images and embeddings are 
            uploaded to the platform.
        dataset_id:
            Identifier of the dataset on the Lightly platform. If 
            dataset_id and token are specified, the images and 
            embeddings are uploaded to the platform.
        upload:
            String to determine whether to upload the full images, 
            thumbnails only, or metadata only.

            Must be one of ['full', 'thumbnails', 'metadata']
        embedding_name:
            Assign the embedding a name in order to identify it on the 
            Lightly platform.

    Examples:
        >>> # upload thumbnails to the Lightly platform
        >>> lightly-upload input_dir=data/ token='123' dataset_id='XYZ'
        >>> 
        >>> # upload full images to the Lightly platform
        >>> lightly-upload input_dir=data/ token='123' dataset_id='XYZ' upload='full'
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

    """
    _upload_cli(cfg)


def entry():
    upload_cli()
