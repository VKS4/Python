import sys
import numpy
import matplotlib
import Functions

# print versions of imported libraries using string formatting
print("Python: {}".format(sys.version))
print("Numpy: {}" .format(numpy.__version__))
print("Plotlib: {}" .format(matplotlib.__version__))

# calling function load_array_x from python file Functions
Functions.load_array_x()
