"""
Itay dali 204711196
David Toubul 342395563
Chen Azulay 201017159
"""

import pandas as pd
from sklearn import preprocessing

from functions import getColumnTitles
import numpy as np
import entropy_based_binning as ebb


def menu():
    menuDict = {}
    menuDict['normalization'] = 0
    menuDict['disc'] = 0
    menuDict['discBins'] = 0
    menuDict['algorithm'] = 0
    normal = int(input('For normalization the dataset enter 1 else enter 0:'))
    if normal == 1:
        menuDict['normalization'] = 1
    disc = int(input('For discretization enter 1 else enter 0:'))
    if disc == 1:
        discType = -1
        while discType < 1 or discType > 2:
            # discType = int(input('For equal depth enter 1, for equal width enter 2, for entropy bases enter 3:'))
            discType = int(input('For equal depth enter 1, for equal width enter 2:'))
        menuDict['disc'] = discType
        numOfBins = 0
        while numOfBins < 1:
            numOfBins = int(input('Enter number of bins:'))
        menuDict['discBins'] = numOfBins
    algo = 0
    while algo < 1 or algo > 6:
        algo = int(input('Choose algorithm: \n1 for naiveBayes \n2 for SKlearn_NaiveBayes \n3 for ID3 \n4 for '
                         'SKlearn_ID3 \n5 for KNNClassifier \n6 for KMeans \n:'))
    menuDict['algorithm'] = algo
    return menuDict


def preProcess(table, structureTextFile, menuDict):
    """

    :param num: number of bins
    :param table: pandas DataFrame
    :param structureTextFile: path of structure file
    :return: pandas DataFrame after discretize
    """

    def numericCol(table, structureTextFile):
        """

        :param table:
        :param structureTextFile: path
        :return: list of column which is numeric by the structure file
        """
        structure = pd.read_csv(structureTextFile, sep=" ", names=['type', 'feature', 'data'])
        column = []
        headers = getColumnTitles(table)
        for i in range(structure.shape[0]):
            if 'NUMERIC' in structure.loc[i]['data']:
                column += [headers[i]]
        return column
    table=pd.read_csv(table)
    numericCol = numericCol(table, structureTextFile)
    table = table.applymap(lambda s: s.lower() if type(s) == str else s)
    table = table.dropna(subset=['class'])  # drop the rows with the NaN values in 'class' column

    # missing values
    for col in table.columns:
        if col not in numericCol:
            table[[col]] = table[[col]].fillna(table.mode().iloc[0])  # fillna with value that appears most often of Each Column
        else:
            table[[col]] = table[[col]].fillna(table.mean().iloc[0])  # fillna with mean of Each Column

    # normalization
    if menuDict['normalization'] == 1:
        for col in numericCol:
            table[col] = preprocessing.minmax_scale(table[col])

    # discretization

    if menuDict['disc'] == 1:  # equal depth
        for col in numericCol:
            table[col] = pd.qcut(table[col], menuDict['discBins'], labels=False,
                                 duplicates='drop')  # Discretize variable into equal-sized buckets
    elif menuDict['disc'] == 2:  # equal width\
        for col in numericCol:
            table[col] = pd.cut(table[col], menuDict['discBins'], labels=False,
                                duplicates='drop')  # Discretize variable into equal-sized buckets
    elif menuDict['disc'] == 3:  # entropy bases
        for col in numericCol:
            array = table[col].to_numpy()  # convert to numpy array for the discretization
            _, A = np.unique(array, return_inverse=True)  # costume the array
            A = A.reshape(array.shape)  # costume the array
            try:
                temp = ebb.bin_array(A, nbins=menuDict['discBins'], axis=0)  # Discretize variable into bins
            except:
                temp = pd.cut(table[col], menuDict['discBins'], labels=False,
                              duplicates='drop')  # default discretization
            table[col] = temp.tolist()  # update the relevant column

    return table
