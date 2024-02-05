import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import Hyperband


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(784,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Tune the alpha parameter (learning rate) using Hyperband
    hp_alpha = hp.Float('alpha', min_value=0.0001, max_value=0.1, sampling='LOG')

    model.compile(optimizer=Adam(learning_rate=hp_alpha),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the Hyperband tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='hyperband_tuning',
    project_name='mnist_alpha_tuning'
)

# Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Create the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hyperparameters)

# Train the model
best_model.fit(x_train, y_train, epochs=10)

# Print the best hyperparameters
print(f"Learning Rate (alpha): {best_hyperparameters.get('alpha')}")

# Evaluate the model
evaluation = best_model.evaluate(x_test, y_test)
print("Test accuracy:", evaluation[1])