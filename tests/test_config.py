from tfcgp.config import Config

def test_config():
    c = Config()
    c.update("cfg/base.yaml")
    assert c.cfg["num_nodes"] == 100
    c.reset()
    assert len(c.functions) == 0
    c.update("cfg/test.yaml")
    assert c.cfg["num_nodes"] == 10

def test_functions():
    c = Config()
    c.update("cfg/test.yaml")
    assert c.arity["tf.add"] == 2
