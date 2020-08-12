
"""
Itay dali 204711196
David Toubul 342395563
Chen Azulay 201017159
"""
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
import joblib
from functions import fit_transforms



def feature(feature, df):
    # Create arrays for the features and the response variable
    y = df[feature]
    x = df.drop(feature, axis=1)
    return x, y

def TestTrainFitPlot(train, test):
    # Setup arrays to store train and test accuracies
    # Split into training and test set
    target_train = train['class']
    inputs_train = train.drop('class', axis='columns')
    target_test = test['class']
    inputs_test = test.drop('class', axis='columns')
    inputs_train = fit_transforms(inputs_train)
    inputs_test = fit_transforms(inputs_test)
    knn = KNeighborsClassifier()
    knn.fit(inputs_train, target_train)
    # save model to file
    filename = 'KNN_model.sav'
    joblib.dump(knn, open(filename, 'wb'))
    # Check Accuracy Score
    print('KNN Accuracy: {}'.format(round(knn.score(inputs_test, target_test), 3)))
    # Enum Loop, accuracy results using range on 'n' values for KNN Classifier
    ''' 
    #if we want to check with many different neighbours and do graph , 
    loop and we put the accuracy test and train in toz disctincts array and after build 
    graph and we can see whats is the best neighbours we need to take 
    
    neighbors = np.arange(1, 20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    for acc, n in enumerate(neighbors):
        # Try KNeighbors with each of 'n' neighbors
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(inputs_train, target_train)
        # Training Accuracy
        train_accuracy[acc] = knn.score(inputs_train, target_train)
        # Testing Accuracy
        test_accuracy[acc] = knn.score(inputs_test, target_test)
    # set plot style
    plt.style.use('ggplot')
    plt.title('KNN Neighbors')
    # Set X-Axis Label
    plt.xlabel('Neighbors\n(#)')
    # Set Y-Axis Label
    plt.ylabel('Accuracy\n(%)', rotation=0, labelpad=35)
    # Place Testing Accuracy
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    # Place Training Accuracy
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    # Append Labels on Testing Accuracy
    for a, b in zip(neighbors, test_accuracy):
        plt.text(a, b, str(round(b, 2)))
    # Add Legend
    plt.legend()
    # Generate Plot
    plt.show()
    # Plotting
    # Set Main Title
    '''

def KNNClassifier(test,train, structFile):
    encode = fit_transforms(train)
    encode_ = fit_transforms(test)
    x = feature("class",encode)
    y = feature("class",encode_)
    TestTrainFitPlot(train,test)




