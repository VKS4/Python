import functions_2D_to_3D


def convert_image_to_greyscale():
    # create an instance of the class ImageToGreyscale
    img_to_greyscale = functions_2D_to_3D.ImageToGreyscale()

    # call the load_image() method
    image = img_to_greyscale.load_image(img_to_greyscale.image_path)

    # call the convert_image_to_greyscale() method
    pixels, width, height, bitmap = functions_2D_to_3D.ImageToGreyscale.convert_image_to_greyscale(image)

    # call method from class ImageToGreyscale that uses for loop to access the data in PixelAccess object
    plot_using_for, processed_image = functions_2D_to_3D.ImageToGreyscale.create_greyscale_plot_using_for_loop(pixels,
                                                                                                               width,
                                                                                                               height)

    # # call method from class ImageToGreyscale that uses numpy array to access the data in PixelAccess object
    # plot_using_numpy_array = functions_2D_to_3D.ImageToGreyscale.create_greyscale_plot_using_numpy(pixels)

    return plot_using_for, processed_image


convert_image_to_greyscale()
