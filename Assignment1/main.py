import os
import sys
import sklearn
import configparser

from matplotlib import pyplot as plt


class Dataset:

    def init(self, name):
        if name == 'mnist':
            pass
        elif name == '':
            pass
        else:
            raise Exception('Unhandled dataset')
        self.instances = {'train': [], 'test': []}
        self.labels = {'train': [], 'test': []}


class Classifier:

    def init(self, type, kwargs):
        if type=='knn':
            self.clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=int(kwargs[0]))
        elif type=='network':
            self.clf = None
        elif type=='rf':
            self.clf = sklearn.ensemble.RandomForestClassifier(n_estimators=int(kwargs[0]))
        else:
            raise Exception('Unhandled Classifier')


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    type = config.get("myParams", "type")
    print(type)