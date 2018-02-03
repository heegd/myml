import myml
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

# Normalize
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = myml.LogisticRegression()
model.train(X_train, y_train, num_iterations=1000, learning_rate=0.1)

# Get the train and test predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# print the results
print('Train accuracy: {0}'.format(accuracy_score(y_train, train_predictions)))
print('Test accuracy: {0}'.format(accuracy_score(y_test, test_predictions)))
