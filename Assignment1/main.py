import os
import sys
import sklearn
import configparser

from keras.datasets import mnist

from matplotlib import pyplot as plt


class Dataset:

    def init(self, name):
        if name == 'mnist':
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
        elif name == '':
            pass
        else:
            raise Exception('Unhandled dataset')
        self.instances = {'train': train_X, 'test': test_X}
        self.labels = {'train': train_y, 'test': test_y}


class Classifier:

    def init(self, type, kwargs):
        if type == 'tree':
            self.clf = sklearn.tree.DecisionTreeClassifier()
        elif type=='network':
            self.clf = None
        elif type == 'boosting':
            self.clf = None
        elif type=='svm':
            self.clf = sklearn.svm.SVC()
        elif type=='knn':
            self.clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=int(kwargs[0]))
        else:
            raise Exception('Unhandled Classifier')


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    type = config.get("myParams", "type")
    name = config.get("myParams", "data")
    print(type)