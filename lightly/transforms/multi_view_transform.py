class MultiViewTransform:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [transform(image) for transform in self.transforms]