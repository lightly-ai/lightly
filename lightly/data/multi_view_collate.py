from typing import List, Tuple
from warnings import warn

import torch
from torch import Tensor


class MultiViewCollate:
    """Collate function that combines views from multiple images into a batch.

    This collate function processes a batch of tuples, where each tuple contains
    multiple views of an image, a label, and a filename. It outputs these as
    separate grouped tensors for easy batch processing.

    Example:
        >>> transform = SimCLRTransform()
        >>> dataset = LightlyDataset(input_dir, transform=transform)
        >>> dataloader = DataLoader(dataset, batch_size=4, collate_fn=MultiViewCollate())
        >>> for views, targets, filenames in dataloader:
        >>>     view0, view1 = views  # each view is a tensor of shape (batch_size, channels, height, width)
    """

    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], Tensor, List[str]]:
        """Turns a batch of (views, label, filename) tuples into a single
        (views, labels, filenames) tuple.

        Args:
            batch:
                The input batch as a list of (views, label, filename) tuples, one for
                each image in the batch. `views` is a list of N view tensors, each
                representing a transformed version of the original image. `label` and
                `filename` are the class label and filename for the corresponding image.

                Example:
                    >>> batch = [
                    >>>     ([img_0_view_0, ..., img_0_view_N], label_0, filename_0),   # image 0
                    >>>     ([img_1_view_0, ..., img_1_view_N], label_1, filename_1),   # image 1
                    >>>     ...
                    >>>     ([img_B_view_0, ..., img_B_view_N], label_B, filename_B),  # image B
                    >>> ]

        Returns:
            A tuple containing:
                - **views**: A list of tensors, where each tensor corresponds to one view
                  of every image in the batch. Tensors are concatenated along the batch
                  dimension.
                - **labels**: A tensor of shape `(batch_size,)` with `torch.long` dtype,
                  containing the labels for all images in the batch.
                - **filenames**: A list of strings containing filenames for all images
                  in the batch.

            Example:
                >>> output = (
                >>>     [
                >>>         Tensor([img_0_view_0, ..., img_B_view_0]),    # view 0
                >>>         Tensor([img_0_view_1, ..., img_B_view_1]),    # view 1
                >>>         ...
                >>>         Tensor([img_0_view_N, ..., img_B_view_N]),    # view N
                >>>     ],
                >>>     torch.tensor([label_0, ..., label_B], dtype=torch.long),
                >>>     [filename_0, ..., filename_B],
                >>> )

        Notes:
            If the input batch is empty, a warning is issued, and an empty tuple
            `([], [], [])` is returned.
        """
        labels: List[int] = []
        fnames: List[str] = []

        if len(batch) == 0:
            warn("MultiViewCollate received empty batch.")
            return [], torch.tensor(labels, dtype=torch.long), fnames

        views: List[List[Tensor]] = [[] for _ in range(len(batch[0][0]))]
        for img, label, fname in batch:
            for i, view in enumerate(img):
                views[i].append(view.unsqueeze(0))
            labels.append(label)
            fnames.append(fname)

        unsqueezed_views = [torch.cat(unsqueezed_view) for unsqueezed_view in views]

        return unsqueezed_views, torch.tensor(labels, dtype=torch.long), fnames
