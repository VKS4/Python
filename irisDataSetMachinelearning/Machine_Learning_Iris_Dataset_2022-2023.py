# check package versions
import sys

from matplotlib import pyplot
from pandas import DataFrame

print('Python: {}'.format(sys.version))
import scipy
print('scipy: {}'.format(scipy.__version__))
import numpy
print('numpy: {}'.format(numpy.__version__))
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('pandas: {}'.format(pandas.__version__))
#import sklearn
#print('sklearn: {}'.format(sklearn.__version__))

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
irisSet: DataFrame | None = pandas.read_csv("iris.csv")

# print how many instances(rows) and attributes(columns) does the data set have
print(irisSet.shape)

# print the first 20 lines of data of all columns
print(irisSet.head(20))

# print statistic information about the dataset
print(irisSet.describe())

# box and whisker plots
irisSet.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
irisSet.hist()
pyplot.show()

# scatter plot matrix
pandas.plotting.scatter_matrix(irisSet)
pyplot.show()

array = irisSet.values
X = array[:, 0:4]
Y = array[:, 4]

def splitTest (X, Y):
    print('Length of array X', len(X))
    print('Length of array Y', len(Y))

    A = len(X)*0.2
    B = len(Y)*0.2

    Aa = len(X) * 0.8
    Bb = len(Y) * 0.8

    learnX = array[0:int(Aa), 0:4]
    learnY = array[0:int(Bb), 4]

    testX = array[0:int(A), 0:4]
    testY = array[0:int(B), 4]

    print('Test X array ', testX)
    print('Test Y array ', testY)

    print('Learn X array ', learnX)
    print('Learn Y array ', learnY)

#     return(learnX, learnY, testX, testY)
#
#     pass
#
# learnX, learnY, testX, testY = splitTest(X, Y)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
