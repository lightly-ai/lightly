# -*- coding: utf-8 -*-
"""**Lightly Download:** Download images from the Lightly platform.

This module contains the entrypoint for the **lightly-download**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import shutil

import hydra
from tqdm import tqdm

import lightly.data as data
from lightly.api import get_samples_by_tag
from lightly.cli._helpers import fix_input_path


def _download_cli(cfg, is_cli_call=True):

    tag_name = cfg['tag_name']
    dataset_id = cfg['dataset_id']
    token = cfg['token']

    if not tag_name:
        print('Please specify a tag name')
        print('For help, try: lightly-download --help')
        return

    if not token or not dataset_id:
        print('Please specify your access token and dataset id')
        print('For help, try: lightly-download --help')
        return

    # get all samples in the queried tag
    samples = get_samples_by_tag(
        tag_name,
        dataset_id,
        token,
        mode='list',
        filenames=None
    )

    # store sample names in a .txt file
    with open(cfg['tag_name'] + '.txt', 'w') as f:
        for item in samples:
            f.write("%s\n" % item)
    msg = 'The list of files in tag {} is stored at: '.format(cfg['tag_name'])
    msg += os.path.join(os.getcwd(), cfg['tag_name'] + '.txt')
    print(msg)

    if cfg['input_dir'] and cfg['output_dir']:
        # "name.jpg" -> "/name.jpg" to prevent bugs like this:
        # "path/to/1234.jpg" ends with both "234.jpg" and "1234.jpg"
        samples = [os.path.join(' ', s)[1:] for s in samples]

        # copy all images from one folder to the other
        input_dir = fix_input_path(cfg['input_dir'])
        output_dir = fix_input_path(cfg['output_dir'])

        dataset = data.LightlyDataset(from_folder=input_dir)
        basenames = dataset.get_filenames()

        source_names = [os.path.join(input_dir, f) for f in basenames]
        target_names = [os.path.join(output_dir, f) for f in basenames]

        # only copy files which are in the tag
        indices = [i for i in range(len(source_names))
                   if any([source_names[i].endswith(s) for s in samples])]

        print(f'Copying files from {input_dir} to {output_dir}.')
        for i in tqdm(indices):
            dirname = os.path.dirname(target_names[i])
            os.makedirs(dirname, exist_ok=True)
            shutil.copy(source_names[i], target_names[i])


@hydra.main(config_path='config', config_name='config')
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
        >>> # copy all files in 'my-tag' to a new directory
        >>> lightly-download token='123' dataset_id='XYZ' tag_name='my-tag' \\
        >>>     input_dir=data/ output_dir=new_data/


    """
    _download_cli(cfg)


def entry():
    download_cli()
