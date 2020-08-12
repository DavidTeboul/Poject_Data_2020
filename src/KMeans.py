"""
Itay dali 204711196
David Toubul 342395563
Chen Azulay 201017159
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from random import randint
import joblib
from functions import Discretize, getColumnTitles
import pandas as pd
#from Evaluation import buildMatrix
from Evaluation import Eval
def makeZero(dataSet):
    """

    @param dataSet: list of data
    @return: list of zero in size of dataSet
    """
    arr = []
    size = len(dataSet)
    for i in range(size):
        arr.append(0)
    return arr


def makeCenterList(data):
    """

    @param data: list of CenterPoint for 1D K-Means
    @return: fixed list of center
    """
    newData = []
    for i in range(len(data)):
        newData.append(int(data[i][0]))
    return newData


def makePoint(data1, data2):
    """

    @param data1: list of X value
    @param data2: list of Y value
    @return: list of XY value
    """
    result = []
    for i in range(len(data1)):
        result.append((data1[i], data2[i]))
    return result


def makeRandomPointList(numOfPoint):
    """
    generate random point 0<=x,y<=100

    @param numOfPoint: number of point to generate
    @return: list of numOfPoint points (x,y)
    """
    data = []
    for i in range(numOfPoint):
        data.append((randint(0, 100), (randint(0, 100))))
    return data


def makeXYlist(data):
    """
    split list of (x,y) to 2 lists
    @param data: list of cordinates (x,y)
    @return XY list
        [0] - list of x
        [1] - list of y
    """
    xList = []
    yList = []
    for i in range(len(data)):
        xList.append(data[i][0])
        yList.append(data[i][1])
    result = [xList, yList]
    return result


# Test
# make single line K-Mean
"""
data: List[int] = [1,2,3,8,9,10,15,16,17]
kmeans = KMeans(n_clusters=3)
kmeans.fit(makePoint(data,makeZero(data)))
center = kmeans.cluster_centers_
cList = makeCenterList(center)
#print(center)


#make data in single line
plt.plot(data,makeZero(data), 'ro' )
plt.plot(cList,makeZero(cList), 'bo')
#print(makeCenterList(center))
plt.xticks(list(range(20)))
plt.show()
"""

# make 2D K-Mean
"""
pList = makeRandomPointList(20)
kmeans = KMeans(n_clusters=6)
kmeans.fit(pList)
center = kmeans.cluster_centers_
xyList = makeXYlist(pList)
plt.scatter(xyList[0],xyList[1])
cList = makeXYlist(center)
plt.scatter(cList[0],cList[1])
plt.show()
"""


def numericCol(table, structureTextFile):
    structure = pd.read_csv(structureTextFile, sep=" ", names=['type', 'feature', 'data'])
    column = []
    headers = getColumnTitles(table)

    for i in range(structure.shape[0]):
        if 'NUMERIC' in structure.loc[i]['data']:
            column += [headers[i]]
    return column


def single_kMean(data, cluster=4):
    """
    works for 1D and 2D data list
    @param data: list of data
    @param cluster: number of cluster, default = 4
    @return: list of K-Mean Cluster
    """
    # if isinstance(data[0],int):
    dimension = 1
    """
    else:
        dimension = 2"""
    cList = []
    kmeans = KMeans(n_clusters=cluster)
    # if dimension==1:#fix 1D bug
    kmeans.fit(makePoint(data, makeZero(data)))  # tranfer to 2D
    """else:
        kmeans.fit(data)"""
    center = kmeans.cluster_centers_  # calc cluster center
    # if dimension==1: #fix 1D bug
    cList = makeCenterList(center)
    """elif dimension==2:
        cList = center"""
    return cList


def makeColDict(columns, cluster):
    """

    :param columns: dict of columns name and class count
    :param cluster : dict of cluster center
    :return: dict of colName : yes no count
    """
    colDict = {}
    tmpDict = {}
    for col in columns:
        for val in cluster[col]:
            tmpDict[val] = {'yes': 0, 'no': 0}
        colDict[col] = tmpDict

    return colDict


def incYes(dict, col, center):
    """

    :param dict: counter dict
    :param col: column name for inc
    :return: adding 1 yo yesClass count
    """
    dict[col][center]['yes'] += 1


def incNo(dict, col, center):
    """

        :param dict: counter dict
        :param col: column name for inc
        :return: adding 1 to noClass count
        """
    dict[col][center]['no'] += 1


def getColList(df, colList):
    lst = []
    for col in colList:
        lst.append(df[col].tolist())
    return lst


def floatToInt(lst):
    """

    :param lst: list of float
    :return: list of int
    """
    newlst = []
    for x in lst:
        newlst.append(int(x))
    return newlst


def fixNumeric(data):
    newData = []
    for lst in data:
        newData.append(floatToInt(lst))
    return newData


takeClosest = lambda num, collection: min(collection, key=lambda x: abs(x - num))  # get closest val


def dictToList(dct, columns):
    newlst = []
    for col in columns:
        newlst.append(dct[col].tolist())
    return newlst


def getClass(classDict, row, colList, kmean):
    '''

    :param classDict: classification dict
    :param row: raw of data
    :param colList: list of column
    :param kmean: k-mean dict
    :return:
    '''
    yes = 0
    no = 0
    for col in colList:
        tmp = takeClosest(row[col], kmean[col])
        if classDict[col][tmp] == 'yes':
            yes += 1
        else:
            no += 1
    if yes > no:
        return 'yes'
    else:
        return 'no'


# testFull

""" 
data: List[int] = [1,2,3,8,9,10,15,16,17]
print(kMean(data,3))
print(kMean(makeRandomPointList(15),4))
"""


def K_MeansClass(test, train, struct):
    """
    check k means for each
    @param train:  cvs file for training the module
    @param test:  cvs file for testing the module
    @param struct: text file of the cvs structure
    @return:
    """
    numOfCluster = (int)
    numOfCluster = 5
    column = numericCol(train, struct)  # get column names
    numOfColumn = len(column)
    #train = train.dropna()  # remove NaN raws
    train = train.reset_index(drop=True)
    numOfRow = len(train)
    numericColList = getColList(train, column)  # list of numeric value
    kMeanDict = {}
    for i in range(numOfColumn):
        kMeanDict[column[i]] = (single_kMean(numericColList[i], numOfCluster))

    yesNoDict = makeColDict(column, kMeanDict)  # init YesNo class counter

    # get valss value for each center
    for i in range(numOfRow):
        for col in column:
            if train['class'][i] == 'yes':
                incYes(yesNoDict, col, takeClosest(train[col][i], kMeanDict[col]))
            else:
                incNo(yesNoDict, col, takeClosest(train[col][i], kMeanDict[col]))

    # classification dict
    classDict = {}
    tmpDict = {}
    for col in column:
        for center in kMeanDict[col]:
            if yesNoDict[col][center]['yes'] > yesNoDict[col][center]['no']:
                tmpDict[center] = 'yes'
            else:
                tmpDict[center] = 'no'
        classDict[col] = tmpDict

    # test file
    #test = test.dropna()
    test = test.reset_index(drop=True)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(test)):
        row = test.loc[i, :]  # getRow
        if getClass(classDict, row, column, kMeanDict) == 'yes':
            if test['class'][i] == 'yes':
                tp += 1
            else:
                fp +=1
        else:
            if test['class'][i] == 'yes':
                fn+=1
            else:
                tn+=1
    Eval(tp,tn,fp,fn)

    filename = 'K-means_model.sav'
    joblib.dump(kMeanDict, filename)
    """for i in range(len(train)):
        row = train.loc[i, :]  # getRow
        if getClass(classDict, row, column, kMeanDict) == test['class'][i]:
            yes += 1
    print("success rate for K-Means in train file is: ", (yes / len(test)) * 100, "%")"""

