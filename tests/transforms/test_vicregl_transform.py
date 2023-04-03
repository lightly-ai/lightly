from PIL import Image

from lightly.transforms.vicregl_transform import VICRegLTransform, VICRegLViewTransform


def test_view_on_pil_image():
    single_view_transform = VICRegLViewTransform()
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 100, 100)


def test_multi_view_on_pil_image():
    multi_view_transform = VICRegLTransform(
        global_crop_size=32,
        local_crop_size=8,
        n_local_views=6,
        global_grid_size=4,
        local_grid_size=2,
    )
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 16  # (2 global crops * 2) + (6 local crops * 2)
    global_views = output[:2]
    local_views = output[2:8]
    global_grids = output[8:10]
    local_grids = output[10:]
    assert all(view.shape == (3, 32, 32) for view in global_views)
    assert all(view.shape == (3, 8, 8) for view in local_views)
    assert all(grid.shape == (4, 4, 2) for grid in global_grids)
    assert all(grid.shape == (2, 2, 2) for grid in local_grids)
