import torch
import lightly.transforms.random_crop_and_flip_with_grid as test_module


def test_location_to_NxN_grid():
    # create a test instance of the Location class
    test_location = test_module.Location(
        left=10,
        top=20,
        width=100,
        height=200,
        image_height=244,
        image_width=244,
        horizontal_flip=True,
        vertical_flip=False,
    )

    # create an instance of the class containing the function
    test_class = test_module.RandomResizedCropAndFlip(grid_size=3)

    # call the function with the test location
    result = test_class.location_to_NxN_grid(test_location)

    # create a tensor representing the expected output
    expected_output = torch.tensor(
        [
            [[126.6667, 53.3333], [76.6667, 53.3333], [26.6667, 53.3333]],
            [[126.6667, 153.3333], [76.6667, 153.3333], [26.6667, 153.3333]],
            [[126.6667, 253.3333], [76.6667, 253.3333], [26.6667, 253.3333]],
        ]
    )
    # check that the function output matches the expected output
    assert torch.allclose(result, expected_output)
