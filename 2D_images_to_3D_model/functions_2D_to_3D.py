import os
from PIL import Image


class ImageToGreyscale:
    def __init__(self):
        # loading up a .jpeg image by calling on a system variable, needs to be set up through OS. The system variable
        # contains path to the image. E.g. C:\Documents\...
        # self. makes a variable an instance variable that can be accessed by other methods in the class
        self.image_path = os.environ.get("IMAGE_PATH")

    @staticmethod
    def load_image(image_path):
        # using PIL library to open the image
        image = Image.open(image_path)

        try:
            # line bellow is used to check if the system variable leads to the image file
            image.show()
        except Exception as e:
            print(f"Error: {e}")

        return image

    @staticmethod
    def convert_image_to_greyscale(image):
        # convert uploaded image to a bit map with greyscale values
        bitmap = image.convert("1")

        # loading up the image converted to a bitmap from main file
        pixels = bitmap.load()

        # obtaining the width and the height of the bitmap
        width, height = bitmap.size

        return pixels, width, height
