""" Bounding Box Utils """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


class BoundingBox:
    """TODO

    """

    def __init__(self, x0: float, y0: float, x1: float, y1: float):

        if x0 > 1 or x1 > 1 or y0 > 1 or y1 > 1 or \
            x0 < 0 or x1 < 0 or y0 < 0 or y1 < 0:
            raise ValueError(
                'Bounding Box Coordinates must be relative to '
                'image width and height but are ({x0}, {y0}, {x1}, {y1}).'
            )

        if x0 > x1:
            raise ValueError(
                'x0 must be smaller than or equal to x1 for bounding box '
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
        """

        """
        return cls(x, y, x + w, y + h)

    @property
    def width(self):
        """

        """
        return self.x1 - self.x0

    @property
    def height(self):
        """

        """
        return self.y1 - self.y0

    @property
    def area(self):
        """

        """
        return self.width * self.height