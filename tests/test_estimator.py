from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfcgp.config import Config
from tfcgp.chromosome import Chromosome
from tfcgp.estimator import model_gen
import numpy as np
import tensorflow as tf
import iris_data

c = Config()
c.update("cfg/test.yaml")
(train_x, train_y), (test_x, test_y) = iris_data.load_data()

my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

batch_size = 100
train_steps = 1000
nin = train_x.shape[1]
nout = 3

ch = Chromosome(nin, nout, batch_size)
ch.random(c)
classifier = tf.estimator.Estimator(
    model_fn=model_gen(ch),
    params={
        'feature_columns': my_feature_columns,
        'hidden_bunits': [10, 10],
        'n_masses': 3,
    })

def test_train():
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps)
    assert True

def test_eval():
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, batch_size))
    print("Eval result: ", eval_result)
    assert eval_result['accuracy'] > 0.0
    assert eval_result['accuracy'] < 1.0
    assert True

def test_predict():
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))
    assert True




