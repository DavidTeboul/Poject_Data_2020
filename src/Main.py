"""
Itay dali 
David Toubul 
Chen Azulay 
"""
import sys

from KMeans import K_MeansClass
from NaiveBayes import naiveBayes
from KNNClassifier import KNNClassifier
from id3 import ID3_algorithm
from id3SKlearn import ID3SKlearn_algorithm
from sklearnNaiveBayes import sklearnNaiveBayes
from PreProcess import menu, preProcess

def algo(i):
    switcher = {
        1: naiveBayes,
        2: sklearnNaiveBayes,
        3: ID3_algorithm,
        4: ID3SKlearn_algorithm,
        5: KNNClassifier,
        6: K_MeansClass
    }
    return switcher[i](test, train, structFile)


# command line interface (CLI) (bonus)
path = sys.argv[1]
structFile = path + '/Structure.txt'
trainFile = path + '/train.csv'
testFile = path + '/test.csv'


menu = menu()
train = preProcess(trainFile, structFile, menu)
train.to_csv(path + '/train_clean.csv')
test = preProcess(testFile, structFile, menu)
train.to_csv(path + '/test_clean.csv')
algo(menu['algorithm'])


