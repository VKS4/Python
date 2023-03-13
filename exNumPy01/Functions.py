import numpy


def load_array_x():
    file = open("DataPythonProject.csv")
    array_x = numpy.genfromtxt(file, dtype="i8", delimiter=",")
    print("printing array_x\n", array_x)

    array_y = abs(array_x)
    print("printing array_y\n", array_y)
