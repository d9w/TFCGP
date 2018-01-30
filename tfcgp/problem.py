import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

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

    def __init__(self):
        data = datasets.load_boston()
        # target = to_categorical(data.target)
        target = np.array([data.target]).T;
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data.data, target)
        self.nin = self.x_train.shape[1]
        self.nout = self.y_train.shape[1]
        self.batch_size = self.x_train.shape[0]

    def run(self, c):
        data = self.x_train[:c.batch_size, :]
        with c.g.as_default():
            out = c.get_tensors()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                return sess.run(out, feed_dict={c.tf_in: data})

    def fit(self, c):
        data = self.x_train[:c.batch_size, :]
        y_true = self.y_train[:c.batch_size, :]
        with c.g.as_default():
            out = c.get_tensors()
            loss = tf.losses.mean_squared_error(labels=y_true, predictions=out)
            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for i in range(10):
                    _, lossv = sess.run((train, loss), feed_dict={c.tf_in: data})
                    print(lossv)
                return lossv
