import pytest
import torch

from lightly.data.ijepa_collate import IJEPAMaskCollator

# IJEPAMaskCollator calls torch.utils.data.default_collate, which was added in
# torch 1.11. The minimal dependency set pins an older version.
DEFAULT_COLLATE_AVAILABLE = hasattr(torch.utils.data, "default_collate")
requires_default_collate = pytest.mark.skipif(
    not DEFAULT_COLLATE_AVAILABLE,
    reason="torch.utils.data.default_collate requires torch>=1.11",
)


def test_init__int_input_size() -> None:
    collator = IJEPAMaskCollator(input_size=224, patch_size=32)
    assert (collator.height, collator.width) == (7, 7)


def test_step__increments() -> None:
    collator = IJEPAMaskCollator()
    assert collator.step() == 0
    assert collator.step() == 1


@requires_default_collate
def test_call() -> None:
    batch_size, npred, nenc = 4, 2, 1
    collator = IJEPAMaskCollator(
        input_size=(224, 224), patch_size=32, npred=npred, nenc=nenc
    )
    batch = [(torch.rand(3, 224, 224), 0) for _ in range(batch_size)]

    collated_batch, masks_enc, masks_pred = collator(batch)

    images, labels = collated_batch
    assert images.shape == (batch_size, 3, 224, 224)
    assert labels.shape == (batch_size,)

    assert len(masks_enc) == nenc
    assert len(masks_pred) == npred
    for mask in list(masks_enc) + list(masks_pred):
        # One mask per image, all truncated to the same number of patches.
        assert mask.shape[0] == batch_size
        assert mask.dtype == torch.int64


@requires_default_collate
def test_call__allow_overlap() -> None:
    collator = IJEPAMaskCollator(
        input_size=(224, 224), patch_size=32, allow_overlap=True
    )
    batch = [(torch.rand(3, 224, 224), 0) for _ in range(2)]

    _, masks_enc, masks_pred = collator(batch)

    assert len(masks_enc) == 1
    assert len(masks_pred) == 2


def test_collate_reexport__warns() -> None:
    with pytest.warns(FutureWarning, match="lightly.data.collate"):
        from lightly.data.collate import IJEPAMaskCollator as reexported

    assert reexported is IJEPAMaskCollator


def test_collate_reexport__unknown_attribute() -> None:
    import lightly.data.collate

    with pytest.raises(AttributeError):
        getattr(lightly.data.collate, "DoesNotExist")
