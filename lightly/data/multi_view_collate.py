import torch
from typing import List, Tuple, Union

class MultiViewCollate:

    def __call__(self, batch: List[Tuple[List[torch.Tensor], int, str]]) -> Tuple[List[torch.Tensor], List[int], List[str]]: 
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch:
                The input batch.

        Returns:
            A tuple containing lists of views, labels and filenames.

        """
        """List[Tuple[List[Tensor(C, H, W)], int, str]]
        Tuple[List[Tensor(B, C, H, W)], List[int], List[str]]"""

        '''
        [
            ([image_0_view_0, image_0_view_1], target_0, filename_0),
            ([image_1_view_0, image_1_view_1], target_1, filename_1),
        ]

        --> 

        (
            [
                Tensor([image_0_view_0, image_1_view_0]),
                Tensor([image_0_view_1, image_1_view_1]),
            ],
            [target_0, target_1],
            [filename_0, filename_1],
        )
        '''

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
