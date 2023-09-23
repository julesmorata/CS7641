import sys
import configparser

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from keras.datasets import mnist, boston_housing
from sklearn.metrics import accuracy_score, confusion_matrix

from matplotlib import pyplot as plt


class Dataset:

    def __init__(self, name):
        if name == 'mnist':
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
            n,w,h = np.shape(train_X)
            train_X = train_X.reshape(n,w*h)
            n,w,h = np.shape(test_X)
            test_X = test_X.reshape(n,w*h)
        elif name == 'boston':
            (train_X, train_y), (test_X, test_y) = boston_housing.load_data()
            train_y = [e//10 for e in train_y]
            test_y = [e//10 for e in test_y]
        else:
            raise Exception('Unhandled dataset')
        self.instances = {'train': train_X, 'test': test_X}
        self.labels = {'train': train_y, 'test': test_y}


class Classifier:

    def __init__(self, kwargs):
        if kwargs["type"] == 'tree':
            self.clf = DecisionTreeClassifier(max_depth=int(kwargs["max_depth"]), min_samples_leaf=int(kwargs["min_samples_leaf"]), max_features=None if kwargs["max_features"]=='None' else int(kwargs["max_features"]), ccp_alpha=float(kwargs["ccp_alpha"]))
        elif kwargs["type"]=='network':
            self.clf = None
        elif kwargs["type"] == 'boosting':
            self.clf = GradientBoostingClassifier(max_depth=int(kwargs["max_depth"]), learning_rate=float(kwargs["lr"]), ccp_alpha=float(kwargs["ccp_alpha"]))
        elif kwargs["type"]=='svm':
            self.clf = SVC(C=float(kwargs["C"]), degree=int(kwargs["deg"]), gamma=float(kwargs["gamma"]))
        elif kwargs["type"]=='knn':
            self.clf = KNeighborsClassifier(n_neighbors=int(kwargs["k"]), kernelweights=kwargs["weights"])
        else:
            raise Exception('Unhandled Classifier')


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    name = config.get("Dataset", "data")
    clfArgs = {}
    for option in config.options("Classifier"):
        clfArgs[option] = config.get("Classifier", option)

    myDataset = Dataset(name)
    myClf = Classifier(clfArgs)

    myClf.clf.fit(myDataset.instances['train'], myDataset.labels['train'])
    train_preds = myClf.clf.predict(myDataset.instances['train'])
    test_preds = myClf.clf.predict(myDataset.instances['test'])
    train_acc = accuracy_score(myDataset.labels['train'], train_preds)
    train_conf_mat = confusion_matrix(myDataset.labels['train'], train_preds)
    test_acc = accuracy_score(myDataset.labels['test'], test_preds)
    test_conf_mat = confusion_matrix(myDataset.labels['test'], test_preds)
    print('training accuracy :', train_acc)
    print('training confusion matrix :', train_conf_mat)
    print('test accuracy :', test_acc)
    print('test confusion matrix :', test_conf_mat)