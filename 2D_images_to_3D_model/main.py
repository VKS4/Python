import os
from PIL import Image

# loading up a .jpeg image by calling on a system variable, needs to be set up through OS. The system variable contains
# path to the image. E.g. C:\Documents\...
image_path = os.environ.get("IMAGE_PATH")

# using PIL library to open the image
image = Image.open(image_path)

# line bellow is used to check if the system variable leads to the image file
# image.show()

# convert uploaded image to a bit map with greyscale values
bitmap = image.convert("1")
