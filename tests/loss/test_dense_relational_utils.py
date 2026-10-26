from __future__ import annotations

import torch

from lightly.loss.dense_relational_utils import roi_resample_to_grid


class TestRoiResampleToGrid:
    def test_output_shape(self) -> None:
        feat = torch.randn(2, 8, 16, 16)
        boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0], [4.0, 4.0, 12.0, 12.0]])
        out = roi_resample_to_grid(feat, boxes, out_h=16, out_w=16)
        assert out.shape == (2, 16 * 16, 8)

    def test_subregion_matches_manual_crop(self) -> None:
        # A box on exact cell boundaries resamples the corresponding sub-grid.
        feat = torch.arange(4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
        # Top-left 2x2 region onto a 2x2 grid.
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        out = roi_resample_to_grid(feat, boxes, out_h=2, out_w=2)
        out_map = out.reshape(1, 2, 2, 1).permute(0, 3, 1, 2)
        assert torch.allclose(out_map, feat[:, :, :2, :2], atol=1e-4)
