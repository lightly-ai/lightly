from typing import List, Tuple, Union
from warnings import warn

import torch
from torch import Tensor


class MultiViewCollate:
    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], Tensor, List[str]]:
        """Turns a batch of tuples into single tuple.

        Args:
            batch:
                The input batch. It is a list of (views, label, filename) tuples for each file in the dataset.
                In particular, views is the output of the augmentation, so it is a list of tensors with n views.
                For example:
                [
                    ([image_0_view_0, image_0_view_1, ...], label_0, filename_0),
                    ([image_1_view_0, image_1_view_1, ...], label_1, filename_1),
                    ...
                ]

        Returns:
            A (views, labels, filenames) tuple. Views is a list of tensors with each tensor containing one
            view for every image in the batch. For example:
            (
                [
                    Tensor([image_0_view_0, image_1_view_0, ...]),    # view 0
                    Tensor([image_0_view_1, image_1_view_1, ...]),    # view 1
                    ...
                ],
                [label_0, label_1, ...],
                [filename_0, filename_1, ...]
            )


        """
        if len(batch) == 0:
            warn("MultiViewCollate received empty batch.")
            return [], [], []

        views = [[] for _ in range(len(batch[0][0]))]
        labels = []
        fnames = []
        for img, label, fname in batch:
            for i, view in enumerate(img):
                views[i].append(view.unsqueeze(0))
            labels.append(label)
            fnames.append(fname)
        for i, view in enumerate(views):
            views[i] = torch.cat(view)

        labels = torch.tensor(
            labels, dtype=torch.long
        )  # Conversion to tensor to ensure backwards compatibility

        return views, labels, fnames
