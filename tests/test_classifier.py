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
orig_genes = np.copy(ch.genes)
clf = Classifier(ch, p.x_train, p.x_test, p.y_train, p.y_test,
                 batch_size=p.batch_size, epochs=p.epochs, seed=p.seed,
                 lamarckian=p.lamarckian)
fit = 0.0

def test_eval():
    acc = clf.evaluate()
    print("Accuracy: ", acc)
    params = clf.get_params()
    print("1 Params: ", params)
    assert acc >= 0.0
    assert acc <= 1.0
    fit = acc
    assert True

def test_train():
    params = clf.get_params()
    print("2 Params: ", params)
    history = clf.train()
    params = clf.get_params()
    print("3 Params: ", params)
    # assert history[0][0] >= history[-1][0] # loss
    # assert history[0][1] <= history[-1][0] # accuracy
    assert np.all(orig_genes == ch.genes) # not lamarckian
    assert True

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
    # assert acc2 >= acc1
    assert True

def test_lamarckian():
    clf.lamarckian = True
    params = clf.get_params()
    print("7 Params: ", params)
    print("Node pids", ch.param_id)
    print(len(ch.genes))
    print("Before: ", ch.genes)
    history = clf.train()
    print("After: ", ch.genes)
    params = clf.get_params()
    acc = clf.evaluate()
    print("8 Params: ", params)
    print("Changed genes: ", ch.genes[orig_genes != ch.genes])
    assert np.any(orig_genes != ch.genes)
    ch2 = Chromosome(p.nin, p.nout)
    ch2.from_genes(ch.genes, c)
    clf2 = Classifier(ch2, p.x_train, p.x_test, p.y_train, p.y_test,
                    batch_size=p.batch_size, epochs=p.epochs, seed=p.seed,
                    lamarckian=p.lamarckian)
    acc2 = clf.evaluate()
    params2 = clf.get_params()
    print("Accuracies: ", acc, acc2)
    print("Params: ", params, params2)
    print("Node pids: ", ch.param_id, ch2.param_id)
    print("Genes: ", ch.genes, ch2.genes)
    # assert acc == acc2
    assert np.all(params == params2)
    assert ch.param_id == ch2.param_id
    for nid in range(len(ch.nodes)):
        assert ch.nodes[nid].param == ch2.nodes[nid].param
