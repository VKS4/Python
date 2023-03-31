import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy
from skimage import measure


class ImageToGreyscale:
    def __init__(self):

        # loading up a .jpeg image by calling on a system variable, needs to be set up through OS. The system variable
        # contains path to the image. E.g. C:\Documents\...
        # self. makes a variable an instance variable that can be accessed by other methods in the class
        self.image_path = os.environ.get("IMAGE_PATH")

    @classmethod
    def load_image(cls, image_path):

        # using PIL library to open the image
        image = Image.open(image_path)

        try:
            # line bellow is used to check if the system variable leads to the image file. If the image path is wrong
            # the program will return exception
            image.show()
        except Exception as e:
            print(f"Error: {e}")

        return image

    @staticmethod
    def convert_image_to_greyscale(image):

        # convert uploaded image to a bit map with greyscale values
        bitmap = image.convert("L")

        # loading up the image converted to a bitmap from main file
        pixels = bitmap.load()

        # obtaining the width and the height of the bitmap
        width, height = bitmap.size

        # print the width and height of the picture
        print(f"width {width}\nheight {height}")

        return pixels, width, height, bitmap

    @staticmethod
    def find_edges_using_PIL(pixels, width, height):

        # using for loop to go through the PixelAccess object
        gray_pixels = [[pixels[x, y] for x in range(width)] for y in range(height)]
        gray_pixels = numpy.array(gray_pixels, dtype=numpy.uint8)
        processed_image = Image.fromarray(gray_pixels)

        plt.suptitle("Greyscale of the uploaded image")
        plt.imshow(gray_pixels, cmap='gray')
        plot_using_for = plt.show()

        processed_image = processed_image.filter(ImageFilter.CONTOUR)
        processed_image.show()

        return plot_using_for, processed_image


    # @staticmethod
    # def find_edges_using_skimage(pixels, width, height):
    #
    #     # using for loop to go through the PixelAccess object
    #     gray_pixels = [[pixels[x, y] for x in range(width)] for y in range(height)]
    #     # gray_pixels = numpy.array(gray_pixels, dtype=numpy.uint8)
    #     processed_image = Image.fromarray(gray_pixels)
    #
    #     plt.suptitle("Greyscale of the uploaded image")
    #     plt.imshow(gray_pixels, cmap='gray')
    #     plot_using_for = plt.show()
    #
    #     processed_image = processed_image.filter(ImageFilter.CONTOUR)
    #     processed_image.show()
    #
    #     return