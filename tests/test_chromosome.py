from tfcgp.config import Config
from tfcgp.chromosome import Chromosome

c = Config()
c.update("cfg/test.yaml")

def test_creation():
    ch = Chromosome(5, 2)
    ch.random(c)
    assert len(ch.nodes) == 15
    assert len(ch.outputs) == 2
