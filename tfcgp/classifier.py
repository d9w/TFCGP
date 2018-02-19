import tensorflow as tf
import numpy as np

class Classifier:

    def __init__(self, chromosome, x_train, x_test, y_train, y_test,
                 batch_size, epochs, seed, lamarckian):

        self.train_g = tf.Graph()
        self.sess = tf.Session(graph = self.train_g)
        # self.test_g = tf.Graph()
        self.chromosome = chromosome

        self.batch_size = batch_size
        self.steps = np.floor(x_train.shape[0]/batch_size).astype(int)
        self.epochs = epochs
        self.lr = 0.1
        self.lamarckian = lamarckian

        with self.train_g.as_default():
            self.chromosome.setup()
            train = tf.data.Dataset.from_tensor_slices((x_train, y_train));
            train = train.shuffle(seed).repeat().batch(self.batch_size)
            train_features, train_labels = train.make_one_shot_iterator().get_next()
            program_out = self.chromosome.get_tensors(train_features, self.batch_size)
            program_out = tf.add(program_out, tf.reduce_min(program_out))
            out_sum = tf.reduce_sum(program_out)
            self.logits = tf.divide(program_out, tf.cond(tf.greater(out_sum, 0),
                                                         lambda: out_sum,
                                                         lambda: tf.constant(1.0)))
            self.loss_op = tf.losses.softmax_cross_entropy(onehot_labels=train_labels,
                                                           logits=self.logits)
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss_op,
                                              global_step=tf.train.get_global_step())
            self.acc_labels = tf.argmax(train_labels, 1)
            self.predicted_classes = tf.argmax(self.logits, 1)
            self.acc_op, self.acc_update = tf.metrics.accuracy(
                labels=self.acc_labels, predictions=self.predicted_classes)
            init = tf.global_variables_initializer()
            self.sess.run(init)

        # with self.test_g.as_default():
        #     self.chromosome.setup()
        #     test = tf.data.Dataset.from_tensor_slices((x_test, y_test));
        #     test = test.shuffle(seed).repeat().batch(self.batch_size)
        #     self.test_itr = test.make_one_shot_iterator().get_next()

    def train(self):
        history = []
        start_acc = self.evaluate()
        history.append([0.0, start_acc])
        with self.train_g.as_default():
            init = tf.local_variables_initializer()
            self.sess.run(init)
            for epoch in range(self.epochs):
                mloss = 0.0
                acc = 0.0
                count = 0.0
                for step in range(self.steps):
                    loss, _, acc, _ = self.sess.run((self.loss_op, self.train_op,
                                                     self.acc_op, self.acc_update))
                    mloss += loss; count += 1
                history.append([mloss/count, acc])
            if self.lamarckian:
                p = self.get_params()
                self.chromosome.set_params(p)
        return history

    def print_params(self):
        for k in self.chromosome.param_id.keys():
            print("Node ", k, ", id ", self.chromosome.param_id[k], ", val ",
                    self.chromosome.nodes[k].param, ", tf ",
                    self.sess.run(self.chromosome.params[self.chromosome.param_id[k]]))

    def get_params(self):
        params = []
        for p in range(len(self.chromosome.params)):
            params += [self.sess.run(self.chromosome.params[p])]
        return params

    def evaluate(self):
        total = 0.0
        count = 0.0
        with self.train_g.as_default():
            for i in range(self.steps):
                labels, pred = self.sess.run((self.acc_labels, self.predicted_classes))
                total += np.sum(labels == pred)
                count += len(labels)
        return total/count
