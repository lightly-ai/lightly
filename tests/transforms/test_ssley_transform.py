from PIL import Image

from lightly.transforms import SSLEYTransform


class TestSSLEYTransform:
    def test__call__(self) -> None:
        transform = SSLEYTransform(input_size=32)

        sample = Image.new("RGB", (100, 100))
        output = transform(sample)
        assert len(output) == 2
        assert all(out.shape == (3, 32, 32) for out in output)
