from tfcgp.config import Config
from tfcgp.chromosome import Chromosome
from tfcgp.classifier import Classifier
from tfcgp.problem import Problem
import numpy as np
import tensorflow as tf
from sklearn import datasets

c = Config()
c.update("cfg/test.yaml")

data = datasets.load_iris()

p = Problem(data.data, data.target)
ch = Chromosome(p.nin, p.nout)
ch.random(c)
clf = Classifier(ch, p.x_train, p.x_test, p.y_train, p.y_test,
                 batch_size=p.batch_size, epochs=p.epochs, seed=p.seed)
fit = 0.0

def test_eval():
    acc = clf.evaluate()
    print("Accuracy: ", acc)
    params = clf.get_params()
    print("1 Params: ", params)
    assert acc >= 0.0
    assert acc <= 1.0
    fit = acc

def test_train():
    params = clf.get_params()
    print("2 Params: ", params)
    history = clf.train()
    params = clf.get_params()
    print("3 Params: ", params)
    assert history[0][0] >= history[-1][0] # loss
    assert history[0][1] <= history[-1][0] # accuracy

def test_improvement():
    print("Test improvement")
    # clf.delete()
    # clf2 = Classifier(ch, p.x_train, p.x_test, p.y_train, p.y_test,
                 # batch_size=p.batch_size, epochs=p.epochs, seed=p.seed)
    params = clf.get_params()
    print("4 Params: ", params)
    acc1 = clf.evaluate()
    params = clf.get_params()
    print("5 Params: ", params)
    history = clf.train()
    params = clf.get_params()
    print("6 Params: ", params)
    acc2 = clf.evaluate()
    print("Trained accuracy: ", acc1, acc2)
    assert acc2 >= acc1
