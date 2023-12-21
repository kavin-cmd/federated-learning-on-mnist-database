import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mnist

# Load or preprocess your data 
(partitioned_x_train, partitioned_y_train), (test_x, test_y) = mnist.load({'#clients': 100, 'mu': 1.5, 'sigma': 3.45})

X = np.random.rand(1000, 784)
y = np.random.randint(0, 10, 1000)

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the RandomForestClassifier 
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None, 1, 2, 3]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
)


# Fit the model to the data
grid_search.fit(train_x, train_y)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test set
predictions = best_model.predict(test_x)

# Evaluate the accuracy
accuracy = accuracy_score(test_y, predictions)
print(f'Best Parameters: {best_params}')
print(f'Test Accuracy: {accuracy:.2%}')
