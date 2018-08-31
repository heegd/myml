import numpy as np

def _sigmoid_function(z):
    g = (1 / (1 + np.exp(-z)))
    return g

def _relu_function(z):
    g = z.copy()
    g[g < 0] = 0
    return g

def _tanh_function(z):
    g = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return g

def _linear_function(x, w, b):
    z = np.dot(x, w) + b
    return z

class LogisticRegression:
    def __init__(self, learning_rate):
        self._w = None
        self._b = None
        self._z = None
        self._a = None
        self.costs = list()
        self.learning_rate = learning_rate

    def _initialize_parameters(self, n_x):
        self._w = np.zeros((n_x, 1))
        self._b = 0

    def _cost_function(self, y):
        m = y.shape[0]
        loss = y * np.log(self.a) + (1 - y) * np.log(1 - self.a)
        j = (-1/m) * np.sum(loss, axis=0)
        return j[0] 

    def _forward_prop(self, x):
        z = _linear_function(x, self._w, self._b)
        self.a = _sigmoid_function(z)

    def _back_prop(self, x, y):
        m = y.shape[0]
        dz = (self.a - y)
        dw = (1 / m) * np.dot(x.T, dz)
        db = (1 / m) * np.sum(dz, axis=0, keepdims=True)
        self._w = self._w - self.learning_rate * dw
        self._b = self._b - self.learning_rate * db

    def train(self, x, y, num_iterations):
        self._initialize_parameters(x.shape[1])
        self.costs = list()

        for i in range(0, num_iterations):
            self._forward_prop(x)
            self.costs.append(self._cost_function(y))
            self._back_prop(x, y)

    def predict(self, x):
        self._forward_prop(x)
        y = self.a > 0.5
        y = y.astype(int)
        return y


class NeuralNetwork:
    class _Layer:
        def __init__(self, num_nodes, w, b, z, a, activation_fn):
            self.num_nodes = num_nodes
            self.w = w
            self.b = b
            self.z = z
            self.a = a
            self.activation_fn = activation_fn

    class Layer:
        def __init__(self, num_nodes, activation_fn):
            self.num_nodes = num_nodes
            self.activation_fn = activation_fn

    def __init__(self, layers, learning_rate):
        self.layers = [self._Layer(num_nodes=l.num_nodes, w=None, b=None, z=None, a=None, activation_fn=l.activation_fn) for l in layers]
        self.learning_rate = learning_rate
        self.costs = None

    def _initialize_parameters(self):
        for layer_index in range(1, len(self.layers)):
            self.layers[layer_index].w = np.random.randn(self.layers[layer_index - 1].num_nodes, self.layers[layer_index].num_nodes) * 0.1
            self.layers[layer_index].b = np.zeros((1, self.layers[layer_index].num_nodes))

    def _cost_function(self, y):
        m = y.shape[0]
        a = self.layers[-1].a
        loss = y * np.log(a) + (1 - y) * np.log(1 - a)
        j = (-1/m) * np.sum(loss, axis=0)
        return j[0]

    def _forward_prop(self, x):
        self.layers[0].a = x
        for layer_index in range(1, len(self.layers)):
            self.layers[layer_index].z = _linear_function(self.layers[layer_index - 1].a, self.layers[layer_index].w, self.layers[layer_index].b)

            if self.layers[layer_index].activation_fn == 'sigmoid':
                self.layers[layer_index].a = _sigmoid_function(self.layers[layer_index].z)
            elif self.layers[layer_index].activation_fn == 'tanh':
                self.layers[layer_index].a = _tanh_function(self.layers[layer_index].z)
            elif self.layers[layer_index].activation_fn == 'relu':
                self.layers[layer_index].a = _relu_function(self.layers[layer_index].z)

    def _back_prop(self, x, y):
        m = y.shape[0]
        a = self.layers[-1].a
        da = (-y/a) + ((1-y)/(1-a))

        for layer_index in reversed(range(1, len(self.layers))):
            activation_fn = self.layers[layer_index].activation_fn
            a = self.layers[layer_index].a
            prev_a = self.layers[layer_index - 1].a
            z = self.layers[layer_index].z
            w = self.layers[layer_index].w

            if activation_fn == 'sigmoid':
                dz = da * a * (1 - a)
            elif activation_fn == 'tanh':
                dz = da * (1 - np.power(a, 2))
            elif activation_fn == 'relu':
                dz = da.copy()
                dz[z <= 0] = 0

            dw = (1/m) * prev_a.T.dot(dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            da = dz.dot(w.T)

            self.layers[layer_index].w = self.layers[layer_index].w - self.learning_rate * dw
            self.layers[layer_index].b = self.layers[layer_index].b - self.learning_rate * db

    def train(self, x, y, num_iterations):

        self._initialize_parameters()
        self.costs = list()

        for i in range(0, num_iterations):
            a = self._forward_prop(x)
            self.costs.append(self._cost_function(y))
            self._back_prop(x, y)

    def predict(self, x):
        self._forward_prop(x)
        y = self.layers[-1].a > 0.5
        y = y.astype(int)
        return y


class KMeans:
    def __init__(self, k=5):
        self.k = k
        self.centroids = None

    def _initialize_centroids(self, n_x,):
        self.centroids = np.random.randn(self.k, n_x) + 1

    def _find_closest_centroids(self, x):
        diffs = np.zeros((x.shape[0], self.k))
        for i in range(self.k):
            diffs[:,i] = np.sum(np.square(x - self.centroids[i]), axis=1)
        return np.argmin(diffs, axis=1).reshape(x.shape[0], 1)

    def _compute_centroids(self, x, groups):
        for i in range(self.k):
            xset = x[groups[groups == 1], :]
            self.centroids[i, 1] = (1 / len(xset)) * np.sum(xset)

    def predict(self, x, num_iterations):
        self._initialize_centroids(x.shape[1])

        for i in range(num_iterations):
            groups = self._find_closest_centroids(x)
            self._compute_centroids(x, groups)

        return groups
