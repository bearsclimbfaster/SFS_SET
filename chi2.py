import numpy as np
import scipy.io
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.metrics import accuracy_score


datasets = ['BASEHOCK', 'COIL20', 'colon', 'Isolet', 'lung', 'madelon']
for dataset in datasets:
    mat = scipy.io.loadmat('datasets/' + dataset + '.mat')
    X = mat['X']
    y = mat['Y']

    #split data in training and testing
    indices=np.arange(X.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    X_train=X[indices[0:int(X.shape[0]*2/3)]]
    y_train=y[indices[0:int(X.shape[0]*2/3)]]
    X_test=X[indices[int(X.shape[0]*2/3):]]
    y_test=y[indices[int(X.shape[0]*2/3):]]

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')

    # Normalize between 0 and 1 (non-negative for chi2)
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    X_train = NormalizeData(X_train)
    X_test = NormalizeData(X_test)

    acc_scores_train = []
    acc_scores_test = []
    num_feat = 100
    for i in range(1,num_feat):
        # selecting features using chi2 similarity
        clf_features = SelectKBest(score_func=chi2, k=i)
        clf_features.fit(X_train, y_train)
        # generate new dataset of only selected features
        X_train_new = clf_features.fit_transform(X_train, y_train)
        X_test_new = clf_features.fit_transform(X_test, y_test)

        # make a SVM classifier to test the test accuracy
        clf = svm.SVC()
        clf.fit(X_train_new, np.ravel(y_train, order='C'))
        y_predictions_train = clf.predict(X_train_new)
        y_predictions_test = clf.predict(X_test_new)
        acc_scores_train.append(accuracy_score(y_train, y_predictions_train))
        acc_scores_test.append(accuracy_score(y_test, y_predictions_test))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(1,num_feat), acc_scores_train, linestyle='--', label='Training accuracy')
    plt.plot(np.arange(1,num_feat), acc_scores_test, linestyle='--', label='Test accuracy')
    plt.legend()
    plt.xlabel('# of selected features')
    plt.ylabel('Test accuracy using SVM')
    plt.title('Classifier accuracy on reduced dataset')
    plt.savefig("Results/accuracy_" + dataset + ".png")
    plt.show()
