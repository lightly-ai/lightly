import glob

from PIL import Image

import lightly

# let's get all jpg filenames from a folder
glob_to_data = "/datasets/clothing-dataset/images/*.jpg"
fnames = glob.glob(glob_to_data)

# load the first two images using pillow
input_images = [Image.open(fname) for fname in fnames[:2]]

# create our transform
transform_simclr = lightly.transforms.SimCLRTransform()

# plot the images
fig = lightly.utils.debug.plot_augmented_images(input_images, transform_simclr)

# let's disable blur
transform_simclr_no_blur = lightly.transforms.SimCLRTransform(gaussian_blur=0.0)
fig = lightly.utils.debug.plot_augmented_images(input_images, transform_simclr_no_blur)

# we can also use the DINO transform instead
transform_dino = lightly.transforms.DINOTransform()
fig = lightly.utils.debug.plot_augmented_images(input_images, transform_dino)
