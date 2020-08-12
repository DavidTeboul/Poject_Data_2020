"""
Itay dali 204711196
David Toubul 342395563
Chen Azulay 201017159
"""

import joblib

from functions import fit_transforms
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
def sklearnNaiveBayes(test, train, structFile):
    model=GaussianNB()
    le=preprocessing.LabelEncoder()

    target_train=train['class']
    inputs_train=train.drop('class',axis='columns')

    target_test=test['class']
    inputs_test=test.drop('class',axis='columns')

    inputs_train=fit_transforms(inputs_train)
    model.fit(inputs_train,target_train)
    # save model to file
    filename = 'NaiveBayesSKlearn_model.sav'
    joblib.dump(model, filename)

    inputs_test=fit_transforms(inputs_test)
    print("sklearnNaiveBayes accuracy:",model.score(inputs_test,target_test),"%")