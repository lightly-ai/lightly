from typing import List, Tuple
from warnings import warn

import torch
from torch import Tensor


class MultiViewCollate:
    """Collate function that combines views from multiple images into a batch.

    Example:
        >>> transform = SimCLRTransform()
        >>> dataset = LightlyDataset(input_dir, transform=transform)
        >>> dataloader = DataLoader(dataset, batch_size=4, collate_fn=MultiViewCollate())
        >>> for views, targets, filenames in dataloader:
        >>>     view0, view1 = views # each view is a tensor of shape (batch_size, channels, height, weidth)
        >>>
    """

    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], Tensor, List[str]]:
        """Turns a batch of (views, label, filename) tuples into single
        (views, labels, filenames) tuple.

        Args:
            batch:
                The input batch as a list of (views, label, filename) tuples, one for
                each image in the batch. In particular, views is a list of N view
                tensors. Every view tensor is a transformed version of the original
                image. Label and filename are the class label and filename of the
                corresponding image.

                Example:
                    >>> batch = [
                    >>>     ([img_0_view_0, ..., img_0_view_N], label_0, filename_0),   # image 0
                    >>>     ([img_1_view_0, ..., img_1_view_N], label_1, filename_1),   # image 1
                    >>>     ...
                    >>>     ([img_B_view_0, ..., img_B_view_N], label_B, filename_B]),  # image B
                    >>> ]

        Returns:
            A (views, labels, filenames) tuple. Views is a list of tensors with each
            tensor containing one view for every image in the batch.

            Example:
                >>> output = (
                >>>     [
                >>>         Tensor([img_0_view_0, ..., img_B_view_0]),    # view 0
                >>>         Tensor([img_0_view_1, ..., img_B_view_1]),    # view 1
                >>>         ...
                >>>         Tensor([img_0_view_N, ..., img_B_view_N]),    # view N
                >>>     ],
                >>>     [label_0, ..., label_B],
                >>>     [filename_0, ..., filename_B],
                >>> )
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
