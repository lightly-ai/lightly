import glob
from PIL import Image
import lightly

# let's get all jpg filenames from a folder
glob_to_data = '/datasets/clothing-dataset/images/*.jpg'
fnames = glob.glob(glob_to_data)

# load the first two images using pillow
input_images = [Image.open(fname) for fname in fnames[:2]]

# create our colalte function
collate_fn_simclr = lightly.data.SimCLRCollateFunction()

# plot the images
fig = lightly.utils.debug.plot_augmented_images(input_images, collate_fn_simclr)

# let's disable blur
collate_fn_simclr_no_blur = lightly.data.SimCLRCollateFunction()
fig = lightly.utils.debug.plot_augmented_images(input_images, collate_fn_simclr_no_blur)

# we can also use the DINO collate function instead
collate_fn_dino = lightly.data.DINOCollateFunction()
fig = lightly.utils.debug.plot_augmented_images(input_images, collate_fn_dino)