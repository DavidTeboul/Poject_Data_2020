'''
Itay dali 204711196
David Toubul 342395563
Chen Azulay 201017159
'''
from sklearn.tree import DecisionTreeClassifier
from functions import fit_transforms
import joblib

def ID3SKlearn_algorithm(test,train,structFile):

    train=fit_transforms(train)
    train_target= train['class']
    train_feature=train.drop('class', axis='columns')

    test=fit_transforms(test)
    test_target= test['class']
    test_feature=test.drop('class', axis='columns')

    tree=DecisionTreeClassifier(criterion='entropy',max_depth=100).fit(train_feature,train_target)

    # save model to file
    filename = 'ID3SKlearn_model.sav'
    joblib.dump(tree, filename)

    prediction = tree.predict(test_feature)

    print("ID3SKlearn_algorithm accuracy is: ",tree.score(test_feature,test_target)*100,"%")
