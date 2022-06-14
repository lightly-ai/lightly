"""The lightly.utils package provides global utility methods.

The io module contains utility to save and load embeddings in a format which is
understood by the Lightly library. With the embeddings_2d module, embeddings
can be transformed to a two-dimensional space for better visualization.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.utils.io import save_embeddings
from lightly.utils.io import load_embeddings
from lightly.utils.io import check_embeddings
from lightly.utils.io import load_embeddings_as_dict
from lightly.utils.io import format_custom_metadata
from lightly.utils.io import save_custom_metadata
from lightly.utils.embeddings_2d import fit_pca
from lightly.utils.benchmarking import BenchmarkModule
from lightly.utils.benchmarking import knn_predict
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.debug import generate_grid_of_augmented_images
from lightly.utils.debug import plot_augmented_images
from lightly.utils.version_compare import version_compare
