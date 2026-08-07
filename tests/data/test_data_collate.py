import random

import pytest
import torch
import torchvision

from lightly.data import (
    BaseCollateFunction,
    ImageCollateFunction,
    MultiCropCollateFunction,
    PIRLCollateFunction,
    SimCLRCollateFunction,
    SwaVCollateFunction,
)
from lightly.data.collate import (
    DINOCollateFunction,
    MAECollateFunction,
    MSNCollateFunction,
    MultiViewCollateFunction,
    VICRegCollateFunction,
    VICRegLCollateFunction,
)
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T


class TestDataCollate:
    def create_batch(self, batch_size=16, seed=0):
        torch.manual_seed(0)
        rnd_images = torchvision.datasets.FakeData(size=batch_size)

        fnames = [f"img_{i}.jpg" for i in range(batch_size)]
        labels = [random.randint(0, 5) for i in range(batch_size)]

        batch = []

        for i in range(batch_size):
            batch.append((rnd_images[i][0], labels[i], fnames[i]))

        return batch

    def test_base_collate(self) -> None:
        batch = self.create_batch()
        transform = T.ToTensor()
        collate = BaseCollateFunction(transform)
        samples, labels, fnames = collate(batch)
        samples0, samples1 = samples

        assert collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    def test_image_collate(self) -> None:
        batch = self.create_batch()
        img_collate = ImageCollateFunction()
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    def test_image_collate_tuple_input_size(self) -> None:
        batch = self.create_batch()
        img_collate = ImageCollateFunction(
            input_size=(32, 32),
        )
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    def test_image_collate_random_rotate(self) -> None:
        batch = self.create_batch()
        img_collate = ImageCollateFunction(rr_prob=1.0, rr_degrees=45.0)
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    def test_image_collate_random_rotate__tuple_degrees(self) -> None:
        batch = self.create_batch()
        img_collate = ImageCollateFunction(rr_prob=1.0, rr_degrees=(-15.0, 45.0))
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    def test_simclr_collate_tuple_input_size(self) -> None:
        batch = self.create_batch()
        img_collate = SimCLRCollateFunction(
            input_size=(32, 32),
        )
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    @pytest.mark.parametrize("low", range(6))
    @pytest.mark.parametrize("high", range(2, 4))
    def test_multi_crop_collate(self, high: int, low: int) -> None:
        batch = self.create_batch()
        multi_crop_collate = MultiCropCollateFunction(
            crop_sizes=[32, 16],
            crop_counts=[high, low],
            crop_min_scales=[0.14, 0.04],
            crop_max_scales=[1.0, 0.14],
            transforms=T.ToTensor(),
        )
        samples, labels, fnames = multi_crop_collate(batch)
        assert multi_crop_collate is not None
        assert len(samples) == low + high
        for i, crop in enumerate(samples):
            if i < high:
                assert crop.shape[-1] == 32
                assert crop.shape[-2] == 32
            else:
                assert crop.shape[-1] == 16
                assert crop.shape[-2] == 16
            assert len(crop) == len(labels)

    def test_swav_collate_init(self) -> None:
        swav_collate = SwaVCollateFunction()

    def test_swav_collate_init_fail(self) -> None:
        with pytest.raises(ValueError):
            SwaVCollateFunction(
                crop_sizes=[1],
                crop_counts=[2, 3],
            )

    def test_multi_view_collate(self) -> None:
        to_tensor = T.ToTensor()
        hflip = T.Compose(
            [
                T.RandomHorizontalFlip(p=1),
                to_tensor,
            ]
        )
        vflip = T.Compose(
            [
                T.RandomVerticalFlip(p=1),
                to_tensor,
            ]
        )
        trans = [to_tensor, hflip, vflip]

        collate_fn = MultiViewCollateFunction(trans)
        batch = self.create_batch()
        imgs = batch[0]
        views, labels, fnames = collate_fn(batch)

        assert len(labels) == len(batch)
        assert len(fnames) == len(batch)
        assert torch.equal(views[0][0], to_tensor(imgs[0]))
        assert torch.equal(views[1][0], hflip(imgs[0]))
        assert torch.equal(views[2][0], vflip(imgs[0]))

    def test_dino_collate_init(self) -> None:
        DINOCollateFunction()

    def test_dino_collate_forward(self) -> None:
        batch = self.create_batch()
        collate_fn = DINOCollateFunction()
        views, labels, fnames = collate_fn(batch)

    def test_mae_collate_init(self) -> None:
        MAECollateFunction()

    def test_mae_collate_forward(self) -> None:
        batch = self.create_batch()
        collate_fn = MAECollateFunction()
        views, labels, fnames = collate_fn(batch)

    def test_pirl_collate_init(self) -> None:
        PIRLCollateFunction()

    def test_pirl_collate_forward_tuple_input_size(self) -> None:
        batch = self.create_batch()
        img_collate = PIRLCollateFunction(
            input_size=(32, 32),
        )
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)

    def test_pirl_collate_forward_n_grid(self) -> None:
        batch = self.create_batch()
        img_collate = PIRLCollateFunction(input_size=32, n_grid=3)
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        assert img_collate is not None
        assert len(samples0) == len(samples1)
        assert len(samples1) == len(labels)
        assert samples1.shape == (16, 9, 3, 10, 10)

    def test_msn_collate_init(self) -> None:
        MSNCollateFunction()

    def test_msn_collate_forward(self) -> None:
        batch = self.create_batch()
        img_collate = MSNCollateFunction(
            random_size=24, focal_size=12, random_views=2, focal_views=10
        )
        views, labels, fnames = img_collate(batch)
        assert len(views) == 2 + 10
        assert len(labels) == len(batch)
        assert len(fnames) == len(batch)
        for view in views[:2]:
            assert view.shape == (16, 3, 24, 24)
        for view in views[2:]:
            assert view.shape == (16, 3, 12, 12)

    def test_vicreg_collate_init(self) -> None:
        VICRegCollateFunction()

    def test_vicreg_collate_forward(self) -> None:
        batch = self.create_batch()
        collate_fn = VICRegCollateFunction()
        views, labels, fnames = collate_fn(batch)

    def test_vicregl_collate_init(self) -> None:
        VICRegLCollateFunction()

    def test_vicregl_collate_forward(self) -> None:
        batch = self.create_batch()
        collate_fn = VICRegLCollateFunction()
        views, labels, fnames = collate_fn(batch)
