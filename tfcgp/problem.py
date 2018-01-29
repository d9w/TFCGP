import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Problem:

    def __init__(self):
        boston = datasets.load_boston()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            boston.data, boston.target)
        self.nin = self.x_train.shape[1]
        self.nout = 1

    def run(self, c):
        data = self.x_train[:c.batch_size, :]
        print(data)
        with c.g.as_default():
            out = c.get_tensors()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                return sess.run(out, feed_dict={c.tf_in: data})
