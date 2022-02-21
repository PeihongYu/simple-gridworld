import numpy as np


class occupancyMeasure:
    def __init__(self, shape):
        self.lambdas = np.zeros(shape)
        self.local_lambdas = np.zeros(shape)

    def update_lambdas(self):
        self.local_lambdas /= self.local_lambdas.sum()
        self.lambdas += self.local_lambdas

    def normalize(self):
        self.lambdas /= self.lambdas.sum()


class stateOccupancyMeasure(occupancyMeasure):
    def __init__(self, state_shape):
        super().__init__(state_shape)

    def count_cur_state(self, state):
        i, j = state
        self.local_lambdas[i, j] += 1


class stateActionOccupancyMeasure(occupancyMeasure):
    def __init__(self, state_shape, action_shape):
        shape = [action_shape] + state_shape
        super().__init__(shape)

    def count_cur_state(self, state, action):
        i, j = state
        self.local_lambdas[action, i, j] += 1
