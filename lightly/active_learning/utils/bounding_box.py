""" Bounding Box Utils """


# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


class BoundingBox:
    """Class which unifies different bounding box formats.

    Attributes:
        x0:
            x0 coordinate (normalized to [0, 1])
        y0:
            y0 coordinate (normalized to [0, 1])
        x1:
            x1 coordinate (normalized to [0, 1])
        y1:
            y1 coordinate (normalized to [0, 1])

    Examples:
    >>> # simple case, format (x0, y0, x1, y1)
    >>> bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
    >>>
    >>> # same bounding box in x, y, w, h format
    >>> bbox = BoundingBox.from_x_y_w_h(0.1, 0.2, 0.2, 0.2)
    >>>
    >>> # often the coordinates are not yet normalized by image size
    >>> # for example, for a 100 x 100 image, the coordinates could be
    >>> # (x0, y0, x1, y1) = (10, 20, 30, 40)
    >>> W, H = 100, 100 # get image shape
    >>> bbox = BoundingBox(10 / W, 20 / H, 30 / W, 40 / H)

    """

    def __init__(self, x0: float, y0: float, x1: float, y1: float, clip_values: bool = False):
        """
            clip_values:
                Set to true to clip the values into [0, 1] instead of raising an error if they lie outside.
        """

        if clip_values:
            def clip_to_0_1(value):
                return min(1, max(0, value))

            x0 = clip_to_0_1(x0)
            y0 = clip_to_0_1(y0)
            x1 = clip_to_0_1(x1)
            y1 = clip_to_0_1(y1)

        if x0 > 1 or x1 > 1 or y0 > 1 or y1 > 1 or \
                x0 < 0 or x1 < 0 or y0 < 0 or y1 < 0:
            raise ValueError(
                f'Bounding Box Coordinates must be relative to '
                f'image width and height but are ({x0}, {y0}, {x1}, {y1}).'
            )

        if x0 > x1:
            raise ValueError(
                f'x0 must be smaller than or equal to x1 for bounding box '
                f'[{x0}, {y0}, {x1}, {y1}]'
            )

        if y0 > y1:
            raise ValueError(
                'y0 must be smaller than or equal to y1 for bounding box '
                f'[{x0}, {y0}, {x1}, {y1}]'
            )

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @classmethod
    def from_x_y_w_h(cls, x: float, y: float, w: float, h: float):
        """Helper to convert from bounding box format with width and height.

        Examples:
        >>> bbox = BoundingBox.from_x_y_w_h(0.1, 0.2, 0.2, 0.2)

        """
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_yolo(cls, x_center: float, y_center: float, w: float, h: float):
        """Helper to convert from yolo format

        Examples:
        >>> bbox = BoundingBox.from_yolo(0.5, 0.4, 0.2, 0.3)

        """
        return cls(x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2, clip_values=True)

    @property
    def width(self):
        """Returns the width of the bounding box relative to the image size.

        """
        return self.x1 - self.x0

    @property
    def height(self):
        """Returns the height of the bounding box relative to the image size.

        """
        return self.y1 - self.y0

    @property
    def area(self):
        """Returns the area of the bounding box relative to the area of the image.

        """
        return self.width * self.height
