""" I/O operations to save and load embeddings. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import json
import csv
from typing import List, Tuple, Dict
import re

import numpy as np

INVALID_FILENAME_CHARACTERS = [',']


def _is_valid_filename(filename: str) -> bool:
    """Returns False if the filename is misformatted.

    """
    for character in INVALID_FILENAME_CHARACTERS:
        if character in filename:
            return False
    return True


def check_filenames(filenames: List[str]):
    """Raises an error if one of the filenames is misformatted

    Args:
        filenames:
            A list of string being filenames

    """
    invalid_filenames = [f for f in filenames if not _is_valid_filename(f)]
    if len(invalid_filenames) > 0:
        raise ValueError(f'Invalid filename(s): {invalid_filenames}')


def check_embeddings(path: str):
    """Raises an error if the embeddings csv file has not the correct format
    
    Use this check whenever you want to upload an embedding to the Lightly 
    Platform.
    This method only checks whether the header row matches the specs:
    https://docs.lightly.ai/getting_started/command_line_tool.html#id1

    Args:
        path:
            Path to the embedding csv file

    Raises:
        RuntimeError
    """
    header = []
    # with open(path, 'r', newline='') as csv_file:
    csv_file = open(path, 'r', newline='')
    reader = csv.reader(csv_file, delimiter=',')
    header: List[str] = next(reader)

    # check for whitespace in the header (we don't allow this)
    if any(x != x.strip() for x in header):
        raise RuntimeError('Embeddings csv file must not contain whitespaces.')

    # first col is `filenames`
    if header[0] != 'filenames':
        raise RuntimeError(f'Embeddings csv file must start with `filenames` '
                        f'column but had {header[0]} instead.')

    # last column is `labels`
    try:
        header_labels_idx = header.index('labels')
    except ValueError:
        raise RuntimeError(f'Embeddings csv file must end with `labels` '
                        f'column but had {header[-1]} instead.')

    # all cols except first and last are `embedding_x`
    for embedding_header in header[1:header_labels_idx]:
        if not re.match(r'embedding_\d+', embedding_header):
            raise RuntimeError(
                f'Embeddings csv file must have `embedding_x` columns but '
                f'found {embedding_header} instead.'
                )
    
    # check for empty rows in the body of the csv file
    for row in reader:
        if len(row) == 0:
            raise RuntimeError('Embeddings csv file must not have empty rows.')
    csv_file.close()

def save_embeddings(path: str,
                    embeddings: np.ndarray,
                    labels: List[int],
                    filenames: List[str]):
    """Saves embeddings in a csv file in a Lightly compatible format.

    Creates a csv file at the location specified by path and saves embeddings,
    labels, and filenames.

    Args:
        path:
            Path to the csv file.
        embeddings:
            Embeddings of the images as a numpy array (n x d).
        labels:
            List of integer labels.
        filenames:
            List of filenames.

    Raises:
        ValueError: If embeddings, labels, and filenames have different lengths.

    Examples:
        >>> import lightly.utils.io as io
        >>> io.save_embeddings(
        >>>     'path/to/my/embeddings.csv',
        >>>     embeddings,
        >>>     labels,
        >>>     filenames)
    """
    check_filenames(filenames)

    n_embeddings = len(embeddings)
    n_filenames = len(filenames)
    n_labels = len(labels)

    if n_embeddings != n_labels or n_filenames != n_labels:
        msg = 'Length of embeddings, labels, and filenames should be equal '
        msg += f' but are not: ({n_embeddings}, {n_filenames}, {n_labels})'
        raise ValueError(msg)

    header = ['filenames']
    header = header + [f'embedding_{i}' for i in range(embeddings.shape[-1])]
    header = header + ['labels']
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        for filename, embedding, label in zip(filenames, embeddings, labels):
            writer.writerow([filename] + list(embedding) + [str(label)])


def load_embeddings(path: str):
    """Loads embeddings from a csv file in a Lightly compatible format.

    Args:
        path:
            Path to the csv file.

    Returns:
        The embeddings as a numpy array, labels as a list of integers, and
        filenames as a list of strings in the order they were saved.

        The embeddings will always be of the Float32 datatype.

    Examples:
        >>> import lightly.utils.io as io
        >>> embeddings, labels, filenames = io.load_embeddings(
        >>>     'path/to/my/embeddings.csv')

    """
    check_embeddings(path)

    filenames, labels = [], []
    embeddings = []
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            # skip header
            if i == 0:
                continue
            #  read filenames and labels
            filenames.append(row[0])
            labels.append(int(row[-1]))
            # read embeddings
            embeddings.append(row[1:-1])

    check_filenames(filenames)

    embeddings = np.array(embeddings).astype(np.float32)
    return embeddings, labels, filenames


def load_embeddings_as_dict(path: str,
                            embedding_name: str = 'default',
                            return_all: bool = False):
    """Loads embeddings from csv and store it in a dictionary for transfer.

    Loads embeddings to a dictionary which can be serialized and sent to the
    Lightly servers. It is recommended that the embedding_name is always
    specified because the Lightly web-app does not allow two embeddings with
    the same name.
    
    Args:
        path:
            Path to the csv file.
        embedding_name:
            Name of the embedding for the platform.
        return_all:
            If true, return embeddings, labels, and filenames, too.

    Returns:
        A dictionary containing the embedding information (see load_embeddings)

    Examples:
        >>> import lightly.utils.io as io
        >>> embedding_dict = io.load_embeddings_as_dict(
        >>>     'path/to/my/embeddings.csv',
        >>>     embedding_name='MyEmbeddings')
        >>>
        >>> result = io.load_embeddings_as_dict(
        >>>     'path/to/my/embeddings.csv',
        >>>     embedding_name='MyEmbeddings',
        >>>     return_all=True)
        >>> embedding_dict, embeddings, labels, filenames = result

    """
    embeddings, labels, filenames = load_embeddings(path)

    # build dictionary
    data = {'embeddingName': embedding_name, 'embeddings': []}
    for embedding, filename, label in zip(embeddings, filenames, labels):
        item = {'fileName': filename,
                'value': embedding.tolist(),
                'label': label}
        data['embeddings'].append(item)

    # return embeddings along with dictionary
    if return_all:
        return data, embeddings, labels, filenames
    else:
        return data


class COCO_ANNOTATION_KEYS:
    """Enum of coco annotation keys complemented with a key for custom metadata.

    """
    # image keys
    images: str = 'images'
    images_id: str = 'id'
    images_filename: str = 'file_name'

    # metadata keys
    custom_metadata: str = 'metadata'
    custom_metadata_image_id: str = 'image_id'


def format_custom_metadata(custom_metadata: List[Tuple[str, Dict]]):
    """Transforms custom metadata into a format which can be handled by Lightly.

    Args:
        custom_metadata:
            List of tuples (filename, metadata) where metadata is a dictionary.

    Returns:
        A dictionary of formatted custom metadata.

    Examples:
        >>> custom_metadata = [
        >>>     ('hello.png', {'number_of_people': 1}),
        >>>     ('world.png', {'number_of_people': 3}),
        >>> ]
        >>> 
        >>> format_custom_metadata(custom_metadata)
        >>> > {
        >>> >   'images': [{'id': 0, 'file_name': 'hello.png'}, {'id': 1, 'file_name': 'world.png'}],
        >>> >   'metadata': [{'image_id': 0, 'number_of_people': 1}, {'image_id': 1, 'number_of_people': 3}]
        >>> > }
    
    """
    formatted = {
        COCO_ANNOTATION_KEYS.images: [],
        COCO_ANNOTATION_KEYS.custom_metadata: [],
    }

    for i, (filename, metadata) in enumerate(custom_metadata):
        formatted[COCO_ANNOTATION_KEYS.images].append({
            COCO_ANNOTATION_KEYS.images_id: i,
            COCO_ANNOTATION_KEYS.images_filename: filename,
        })
        formatted[COCO_ANNOTATION_KEYS.custom_metadata].append({
            COCO_ANNOTATION_KEYS.custom_metadata_image_id: i,
            **metadata,
        })

    return formatted


def save_custom_metadata(path: str, custom_metadata: List[Tuple[str, Dict]]):
    """Saves custom metadata in a .json.

    Args:
        path:
            Filename of the .json file where the data should be stored.
        custom_metadata:
            List of tuples (filename, metadata) where metadata is a dictionary.
    
    """
    formatted = format_custom_metadata(custom_metadata)
    with open(path, 'w') as f:
        json.dump(formatted, f)
