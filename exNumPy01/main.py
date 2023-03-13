import sys
import numpy
import pandas
import matplotlib
import Functions

print('Python: {}'.format(sys.version))
print('Numpy: {}' .format(numpy.__version__))
print('Pandas: {}' .format(pandas.__version__))
print('Plotlib: {}' .format(matplotlib.__version__))

Functions.load_array_x()
