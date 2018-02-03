import numpy as np

def _sigmoid_function(z):
    g = (1 / (1 + np.exp(-z)))
    return g

def _linear_function(x, w, b):
    z = np.dot(x, w) + b
    return z

class LogisticRegression:
    def __init__(self):
        self.w = 0
        self.b = 0
        self.costs = list()

    def _initialize_parameters(self, n_x):
        self.w = np.zeros((n_x, 1))
        self.b = 0

    def _cost_function(self, y, a):
        m = y.shape[0]
        loss = y * np.log(a) + (1 - y) * np.log(1 - a)
        j = (-1/m) * np.sum(loss, axis=0)
        return j[0] 

    def _forward_prop(self, x):
        z = _linear_function(x, self.w, self.b)
        a = _sigmoid_function(z)
        return a

    def _back_prop(self, x, y, a):
        m = y.shape[0]
        dz = (a - y)
        dw = (1 / m) * np.dot(x.T, dz)
        db = (1 / m) * np.sum(dz, axis=0, keepdims=True)
        return dw, db

    def train(self, x, y, num_iterations, learning_rate):
        self._initialize_parameters(x.shape[1])
        self.costs = list()

        for i in range(0, num_iterations):
            # forward prop
            a = self._forward_prop(x)

            # get cost
            self.costs.append(self._cost_function(y, a))

            # back prop
            dw, db = self._back_prop(x, y, a)

            # set parameters
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

    def predict(self, x):
        a = self._forward_prop(x)
        y = a > 0.5
        y = y.astype(int)
        return y
