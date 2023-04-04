import sys
import numpy
import matplotlib
import Functions


def print_library_versions():
    # print versions of imported libraries using string formatting
    print("Python: {}".format(sys.version))
    print("Numpy: {}".format(numpy.__version__))
    print("Plotlib: {}".format(matplotlib.__version__))


pass

print_library_versions()
# calling function load_array_x from python file Functions
Functions.load_array_x()
