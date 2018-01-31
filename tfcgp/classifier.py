import tensorflow as tf
import numpy as np

class Classifier:

    def __init__(self, chromosome, x_train, x_test, y_train, y_test,
                 batch_size, epochs, seed):

        self.train_g = tf.Graph()
        self.sess = tf.Session(graph = self.train_g)
        # self.test_g = tf.Graph()
        self.chromosome = chromosome

        self.batch_size = batch_size
        self.steps = np.floor(x_train.shape[0]/batch_size).astype(int)
        self.epochs = epochs
        self.lr = 0.1

        with self.train_g.as_default():
            self.chromosome.setup()
            train = tf.data.Dataset.from_tensor_slices((x_train, y_train));
            train = train.shuffle(seed).repeat().batch(self.batch_size)
            train_features, train_labels = train.make_one_shot_iterator().get_next()
            program_out = self.chromosome.get_tensors(train_features)
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
            self.acc_op, self.acc_update = tf.metrics.accuracy(labels=self.acc_labels,
                                                               predictions=self.predicted_classes)
        # with self.test_g.as_default():
        #     self.chromosome.setup()
        #     test = tf.data.Dataset.from_tensor_slices((x_test, y_test));
        #     test = test.shuffle(seed).repeat().batch(self.batch_size)
        #     self.test_itr = test.make_one_shot_iterator().get_next()

    def train(self):
        history = []
        with self.train_g.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.print_params()
            init = tf.local_variables_initializer()
            self.sess.run(init)
            for i in range(self.steps * self.epochs):
                logits, loss, _, acc, _ = self.sess.run((self.logits, self.loss_op,
                                                         self.train_op,
                                                         self.acc_op, self.acc_update))
                history.append([loss, acc])
            self.print_params()
        return history

    def print_params(self):
        for k in self.chromosome.param_id.keys():
            print("Node ", k, ", id ", self.chromosome.param_id[k], ", val ",
                    self.chromosome.nodes[k].param, ", tf ",
                    self.sess.run(self.chromosome.params[self.chromosome.param_id[k]]))


    def evaluate(self):
        total = 0.0
        count = 0.0
        with self.train_g.as_default():
            for i in range(self.steps):
                labels, pred = self.sess.run((self.acc_labels, self.predicted_classes))
                total += np.sum(labels == pred)
                count += len(labels)
        return total/count

    def test(self):
        pass

    # def run(self, c):
    #     data = self.x_train[:c.batch_size, :]
    #     with c.g.as_default():
    #         out = c.get_tensors()
    #         init = tf.global_variables_initializer()
    #         with tf.Session() as sess:
    #             sess.run(init)
    #             return sess.run(out, feed_dict={c.tf_in: data})

    # def fit(self, c):
    #     with c.g.as_default():
    #         out = c.get_tensors()
    #         labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.nout))
    #         loss = tf.losses.mean_squared_error(labels=labels, predictions=out)
    #         optimizer = tf.train.AdamOptimizer()
    #         train = optimizer.minimize(loss)
    #         init = tf.global_variables_initializer()
    #         with tf.Session() as sess:
    #             sess.run(init)
    #             for i in range(self.epochs):
    #                 d, l = self.train_itr.get_next()
    #                 loss = tf.losses.mean_squared_error(labels=labels, predictions=out)
    #                 _, lossv = sess.run((train, loss), feed_dict={c.tf_in: d, labels: l})
    #                 print(lossv)
    #             return lossv
