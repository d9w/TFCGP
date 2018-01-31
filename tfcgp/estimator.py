import tensorflow as tf
import numpy as np

def model_gen(chromosome):

    def my_model(features, labels, mode, params):
        """DNN with three hidden layers, and dropout of 0.1 probability."""
        # Create three fully connected layers each layer having a dropout
        # probability of 0.1.
        print("Chromosome: ", chromosome)
        net = tf.feature_column.input_layer(features, params['feature_columns'])
        for units in params['hidden_bunits']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        # Compute logits (1 per class).
        logits = tf.layers.dense(net, params['n_masses'], activation=None)

        # Compute predictions.
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Compute loss.
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                    predictions=predicted_classes,
                                    name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return my_model
