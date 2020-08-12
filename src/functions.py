'''
Itay dali 204711196
David Toubul 342395563
Chen Azulay 201017159
'''
import pandas as pd
from sklearn import preprocessing


def getColumnTitles(table):
    """

    :param table: pandas DataFrame
    :return: list of columns
    """
    return list(table.columns)

def Discretize(num, table, structureTextFile):
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
    column = numericCol(table, structureTextFile)
    table = table.applymap(lambda s: s.lower() if type(s) == str else s)
    table=table.dropna(subset=['class'])#drop the rows with the NaN values in 'class' column
    for col in column:
        table[col] = pd.qcut(table[col], num, labels=False, duplicates='drop')#Discretize variable into equal-sized buckets
    table = table.fillna(table.mode().iloc[0])#fillna with value that appears most often of Each Column
    # table.fillna(-1, inplace=True)
    # table.apply(lambda x: x.astype(str).str.lower())
    #table = table.applymap(lambda s: s.lower() if type(s) == str else s)
    return table


def p_xy(table,column_x,value_x,column_class,value_class):
    """

    :param table:
    :param column_x:
    :param value_x:
    :param column_class:
    :param value_class:
    :return:
    """
    length=table[column_x].value_counts()[value_x]
    try:
        p=table.loc[table[column_x]==value_x][column_class].value_counts()[value_class]
    except:
        p=1
    p=round(p/length,3)
    return p

def valuesType(table,column):
    """

    :param table:
    :param column:
    :return: return a list with the name of values in column
    """
    columnValues=table[column].unique().tolist()
    if -1 in columnValues:
        columnValues.remove(-1)
    return columnValues

def pArrayByFeature(table,featureCol,classValue,classCol):
    """
    calculate p(x|y=value)
    :param table:
    :param featureCol:
    :param classValue:y
    :param classCol:
    :return: array of probabilities of the values in the column by each value in 'class' column
    """
    array=[]
    for i in valuesType(table,featureCol):
        array+=[p_xy(table,featureCol,i,classCol,classValue)]
    return array

def fit_transforms(table):
    """

    :param table: pandas DataFrame
    :return: turn string values to int
    """
    le=preprocessing.LabelEncoder()
    columns = getColumnTitles(table)
    for col in columns:
        try:
            table[col] = le.fit_transform(table[col])
        except:
            continue
    return table

