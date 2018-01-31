import tensorflow as tf
import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tfcgp.classifier import Classifier

# from keras
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

class Problem:

    def __init__(self, data, target, logname='test', generation=0,
                 batch_size=32, epochs=100, seed=1234, root_dir='.'):
        # if len(target.shape) == 1:
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
        self.logname = logname
        self.generation = generation
        self.logfile = os.path.join(root_dir, 'logs', logname+'.log')

    def get_fitness(self, chromosome):
        clf = Classifier(chromosome, self.x_train, self.x_test, self.y_train, self.y_test,
                         batch_size=self.batch_size, epochs=self.epochs, seed=self.seed)
        history = clf.train()
        print(history)
        train_acc = clf.evaluate()
        # test_acc = clf.test()
        # with open(self.logfile, 'a') as f:
            # f.write('L,%s,%d,%')
        return train_acc

class Deprecated:

    def __init__(self, batch_size=32, epochs=5, seed=1234):
        self.batch_size = batch_size
        self.epochs = epochs
        data = datasets.load_boston()
        # target = to_categorical(data.target)
        target = np.array([data.target]).T;
        x_train, x_test, y_train, y_test = train_test_split(
            data.data, target)
        self.train = tf.data.Dataset.from_tensor_slices((x_train, y_train));
        self.train = self.train.shuffle(seed).repeat().batch(self.batch_size)
        self.train_itr = self.train.make_one_shot_iterator()
        self.test = tf.data.Dataset.from_tensor_slices((x_test, y_test));
        self.test = self.test.shuffle(seed).repeat().batch(self.batch_size)
        self.test_itr = self.test.make_one_shot_iterator()
        self.nin = x_train.shape[1]
        self.nout = y_train.shape[1]

    def run(self, c):
        data = self.x_train[:c.batch_size, :]
        with c.g.as_default():
            out = c.get_tensors()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                return sess.run(out, feed_dict={c.tf_in: data})

    def fit(self, c):
        with c.g.as_default():
            out = c.get_tensors()
            labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.nout))
            loss = tf.losses.mean_squared_error(labels=labels, predictions=out)
            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for i in range(self.epochs):
                    d, l = self.train_itr.get_next()
                    loss = tf.losses.mean_squared_error(labels=labels, predictions=out)
                    _, lossv = sess.run((train, loss), feed_dict={c.tf_in: d, labels: l})
                    print(lossv)
                return lossv
