import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, args=None):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
    def info(self, agent, world):
        return {}