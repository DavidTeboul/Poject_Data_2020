import pandas as pd


def Eval(tp, tn, fp, fn):
    def fixStr(val, name):
        tmp = name.upper() + " = " + str(val)
        return tmp
    try:
        accuracy = (tp + tn) / (tp + tn + fn + fp)
    except ZeroDivisionError:
        accuracy=0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall=0
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision=0
    try:
        fMeasure = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        fMeasure=0


    print("Evaluation Indicators:")
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("fMeasure: ", fMeasure)
    print()
    data = {'Negative': [fixStr(tn, 'tn'), fixStr(fn, 'fn')], 'Positive': [fixStr(fp, 'fp'), fixStr(tp, 'tp')]}
    matrix = pd.DataFrame(data)
    print("Confusion matrix:")
    print()
    print(matrix)
    print()
