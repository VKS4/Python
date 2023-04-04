# import numpy library
import numpy


# define a function load_array_x
def load_array_x():
    # csv file with name DataPythonProject.csv
    file = open("DataPythonProject.csv")

    # using function genfromtxt from the numpy library and change the data type of the array from original float to int8
    array_x = numpy.genfromtxt(file, dtype="i8", delimiter=",")

    # print array_x after initial message
    print("printing array_x \n {}" .format(array_x))

    # use abs() function to convert any negative numbers to positive values
    array_y = abs(array_x)

    # print the resulting array_y
    print("printing array_y \n {}" .format(array_y))

    # access the first row and first column of array_y and store in a variable element
    element = array_y[0][0]

    # print the variable element after initial message
    print("printing 1st row and a first column of array_y \n {}" .format(element))

    # access several cells of an array_y
    element_range_a = array_y[3][:]

    # printing variable element_range_a
    print("printing 4th row and all columns of array_y \n {}" .format(element_range_a))

    # access first 4 columns of row 5 in array_y
    element_range_b = array_y[4][:4]

    # print variable element_range_b
    print("printing first four columns in the 5th row of array_y \n {}" .format(element_range_b))

    # # access all data between 4th row and 4th column of array_y
    # element_range_c = array_y[:4][:]
    # element_range_c = array_y[:][:4]
    #
    # # print the variable element_range_c
    # print("printing section of array_y \n {}" .format(element_range_c))

    pass
