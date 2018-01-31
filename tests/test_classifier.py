from tfcgp.config import Config
from tfcgp.chromosome import Chromosome
from tfcgp.classifier import Classifier
import numpy as np
import tensorflow as tf
from sklearn import datasets

c = Config()
c.update("cfg/test.yaml")

clf = Classifier()

def test_train():
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps)
    print("After training: ", tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope="program"))
    ch.print_params()
    g = tf.get_default_graph()
    print("3 Default graph: ", g.get_operations())

    assert True

def test_eval():
    eval_result = classifier.evaluate(
        input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, batch_size))
    ch.print_params()
    print("Eval result: ", eval_result)
    assert eval_result['accuracy'] > 0.0
    assert eval_result['accuracy'] < 1.0
    assert True

# def test_predict():
#     expected = ['Setosa', 'Versicolor', 'Virginica']
#     predict_x = {
#         'SepalLength': [5.1, 5.9, 6.9],
#         'SepalWidth': [3.3, 3.0, 3.1],
#         'PetalLength': [1.7, 4.2, 5.4],
#         'PetalWidth': [0.5, 1.5, 2.1],
#     }

#     predictions = classifier.predict(
#         input_fn=lambda:iris_data.eval_input_fn(predict_x,
#                                                 labels=None,
#                                                 batch_size=batch_size))

#     for pred_dict, expec in zip(predictions, expected):
#         template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

#         class_id = pred_dict['class_ids'][0]
#         probability = pred_dict['probabilities'][class_id]

#         print(template.format(iris_data.SPECIES[class_id],
#                               100 * probability, expec))
#     assert True
