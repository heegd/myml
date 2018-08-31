import myml
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Download
X, y = load_breast_cancer(return_X_y=True)

# Reshape
y = y.reshape(y.shape[0], 1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('--------------------------------------------------')
print('Data')
print('--------------------------------------------------')
print('X_train: {0}'.format(X_train.shape))
print('X_test: {0}.'.format(X_test.shape))
print('y_train: {0}.'.format(y_train.shape))
print('y_test: {0}.'.format(y_test.shape))
print()
print()

# Normalize
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic model
print('--------------------------------------------------')
print('Logistic Regression')
print('--------------------------------------------------')
model = myml.LogisticRegression(learning_rate=0.1)
model.train(X_train, y_train, num_iterations=100)

# Get the train and test predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# print the results
print('Logistic Regression Cost {0}:'.format(model.costs[-1]))
print('Logistic Train accuracy: {0}'.format(accuracy_score(y_train, train_predictions)))
print('Logistic Test accuracy: {0}'.format(accuracy_score(y_test, test_predictions)))
print()
print()

# Train the neural network model
print('--------------------------------------------------')
print('Neural Network')
print('--------------------------------------------------')
layers = [myml.NeuralNetwork.Layer(num_nodes=30, activation_fn='relu'),
          myml.NeuralNetwork.Layer(num_nodes=10, activation_fn='relu'),
          myml.NeuralNetwork.Layer(num_nodes=3, activation_fn='relu'),
          myml.NeuralNetwork.Layer(num_nodes=1, activation_fn='sigmoid')]
model = myml.NeuralNetwork(learning_rate=0.1, layers=layers)
model.train(X_train, y_train, num_iterations=1000)

# Get the train and test predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# print the results
print('Neural Network Cost: {0}'.format(model.costs[-1]))
print('Neural Network Train accuracy: {0}'.format(accuracy_score(y_train, train_predictions)))
print('Neural Network Test accuracy: {0}'.format(accuracy_score(y_test, test_predictions)))

# Train the KMeans model
print('--------------------------------------------------')
print('KMeans')
print('--------------------------------------------------')
model = myml.KMeans(k=2)
groups = model.predict(X_train, 1000)
print(np.unique(groups, return_counts=True))
