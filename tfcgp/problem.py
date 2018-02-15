import tensorflow as tf
import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tfcgp.classifier import Classifier

# from keras
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

class Problem:

    def __init__(self, data, target, learn=True, batch_size=32, epochs=100, seed=1234,
                 lamarckian=False):
        if len(target.shape) == 1:
            # target = np.array([target]).T;
            target = to_categorical(target)
        self.nin = data.shape[1]
        self.nout = target.shape[1]
        data_mins = np.min(data, axis=0)
        data_maxs = np.max(data, axis=0)
        data = (data - data_mins) / (data_maxs - data_mins)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, target)
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.learn = learn
        self.lamarckian = lamarckian
        self.eval_count = 0

    def get_fitness(self, chromosome):
        self.eval_count += 1
        clf = Classifier(chromosome, self.x_train, self.x_test, self.y_train, self.y_test,
                         batch_size=self.batch_size, epochs=self.epochs, seed=self.seed,
                         lamarckian=self.lamarckian)
        history = []
        if self.learn:
            history = clf.train()
        train_acc = clf.evaluate()
        clf.sess.close()
        del clf
        return train_acc, history
