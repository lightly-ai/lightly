import torch
from torch import Tensor
from typing import List, Tuple, Union
from warnings import warn


class MultiViewCollate:
    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], List[int], List[str]]:
        """Turns a batch of tuples into a tuple of batches.

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
            A tuple containing lists of images, labels and filenames.
            Structure example:
            (
                [
                    Tensor([image_0_view_0, image_1_view_0]),
                    Tensor([image_0_view_1, image_1_view_1]),
                    ...
                ],
                [label_0, label_1, ...],
                [filename_0, filename_1, ...]
            )


        """
        if len(batch) == 0:
            warn("The batch is empty. Collate returned empty lists.")
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

        return views, labels, fnames
