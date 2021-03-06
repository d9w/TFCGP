from tfcgp.config import Config
from tfcgp.chromosome import Chromosome
import numpy as np
import tensorflow as tf

c = Config()
c.update("cfg/test.yaml")

def test_creation():
    ch = Chromosome(5, 2)
    ch.random(c)
    assert len(ch.nodes) == c.cfg["num_nodes"] + 5
    assert len(ch.outputs) == 2

def test_active():
    ch = Chromosome(5, 2)
    ch.random(c)
    ch.nodes[5].x = 0
    ch.nodes[5].y = 1
    ch.nodes[5].arity = 2
    ch.nodes[6].x = 3
    ch.nodes[7].x = 5
    ch.nodes[7].y = 6
    ch.nodes[7].arity = 1
    ch.outputs[0] = 7
    ch.outputs[1] = 7
    ch.set_active()
    print(ch.get_active())
    assert ch.get_active() == [0, 1, 5, 7]

def test_tensor():
    ch = Chromosome(5, 2)
    ch.random(c)
    ch.outputs[0] = 9
    ch.outputs[1] = 9
    out = ch.get_tensors()
    print(out)
    assert True

def test_visul():
    ch = Chromosome(8, 8)
    ch.random(c)
    ch.visualize("test")
    assert True

def test_run():
    ch = Chromosome(2, 1)
    ch.random(c)
    ch.outputs[0] = 2
    ch.nodes[2].x = 0
    ch.nodes[2].y = 1
    ch.nodes[2].function = tf.square
    ch.nodes[2].arity = 1
    outv = ch.run()
    print(outv)
    assert outv == ch.nodes[2].param

def test_multiout_run():
    ch = Chromosome(10, 10)
    ch.random(c)
    outv = ch.run()
    print(outv)
    assert len(outv) > 1
