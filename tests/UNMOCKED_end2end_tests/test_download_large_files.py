import time

import lightly
lightly.api.utils.RETRY_MAX_RETRIES = 1

from lightly.api.download import download_image

url_33MB = "https://cdn.eso.org/images/original/potw1130a.tif"
url_5MB = "https://cdn.eso.org/images/large/potw1130a.jpg"
url_1_5MB = "https://cdn.eso.org/images/publicationjpg/potw1130a.jpg"

start = time.time()
img = download_image(url_5MB)
print(f"Took {time.time()-start:5.2f}s to download the image.")

img.show()